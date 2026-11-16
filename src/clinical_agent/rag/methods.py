"""
The published methods we compare against. Every method is a small class used the same way:

    method = MedRAG(llm, retriever)
    result = method(question)        # -> {"answer": "B", "trace": {...}}

"answer" is an option letter, or None if the model never clearly picked one.
"trace" keeps the in-between steps, so any answer can be checked afterwards.

    name           what it does                                         LLM calls per question
    -------------  ---------------------------------------------------  ----------------------
    cot            answer once, no library                              1
    cot_sc         answer 5 times, no library, take the majority        1 request (5 answers)
    medrag         whole question -> MedCPT top-32 passages -> answer   1
    rationale_rag  write a short rationale, search with it, answer      2
    imedrag        ask itself follow-up questions over 3 rounds         about 10

Our own method, the diagnostic agent, lives in agent.py.
"""

from collections import Counter

from ..llm import parse_choice, parse_json
from . import prompts


class MCQ:
    """One multiple-choice question."""

    def __init__(self, id, question, options, answer="", dataset=""):
        self.id = id
        self.question = question
        self.options = options     # {"A": "...", "B": "...", ...}
        self.answer = answer       # the correct letter - methods never look at it
        self.dataset = dataset

    def options_text(self):
        lines = []
        for letter in sorted(self.options):
            lines.append(letter + ". " + self.options[letter])
        return "\n".join(lines)


def format_evidence(evidence):
    """The passages as MedRAG's prompt shows them:
        Document [1] (statpearls: Pulmonary Embolism -- Evaluation) text..."""
    if not evidence:
        return "(no evidence retrieved)"
    lines = []
    number = 1
    for item in evidence:
        lines.append(f"Document [{number}] ({item.source}: {item.title}) {item.text[:1500]}")
        number += 1
    return "\n".join(lines)


def evidence_trace(evidence):
    """The short form of the passages that gets saved in the result files."""
    saved = []
    for item in evidence:
        saved.append({"id": item.snippet.id, "score": round(item.score, 3)})
    return saved


# ========================================================================================

class CoT:
    """Chain of thought with no library, using MedRAG's exact prompt."""
    name = "cot"

    def __init__(self, llm, retriever=None):
        self.llm = llm

    def __call__(self, q):
        prompt = prompts.COT_USER.format(question=q.question, options=q.options_text())
        output = self.llm.chat(prompt, system=prompts.COT_SYSTEM, temperature=0.0)
        return {"answer": parse_choice(output, q.options), "trace": {"output": output}}


def majority(votes):
    """The most common real letter. A tie goes to the later letter, just so it's always the same."""
    best = None
    for letter in votes:
        if letter is None:
            continue
        if best is None:
            best = letter
        elif votes[letter] > votes[best]:
            best = letter
        elif votes[letter] == votes[best] and letter > best:
            best = letter
    return best


class SelfConsistency:
    """Answer the same question 5 times with a bit of randomness, and take the majority
    (Wang et al., ICLR 2023). How strongly the 5 answers agree is also a free confidence signal."""
    name = "cot_sc"

    def __init__(self, llm, retriever=None, n=5, temperature=0.7):
        self.llm = llm
        self.n = n
        self.temperature = temperature

    def __call__(self, q):
        prompt = prompts.COT_USER.format(question=q.question, options=q.options_text())
        outputs = self.llm.chat(prompt, system=prompts.COT_SYSTEM, temperature=self.temperature,
                                n=self.n, seed=0)
        if not isinstance(outputs, list):
            outputs = [outputs]

        votes = Counter()                    # e.g. {"B": 3, "A": 2}
        for text in outputs:
            votes[parse_choice(text, q.options)] += 1

        return {"answer": majority(votes), "trace": {"votes": dict(votes)}}


class MedRAG:
    """Plain RAG as in the MIRAGE paper: the whole question is the search query, MedCPT finds
    the top 32 passages, and they are pasted into MedRAG's prompt."""
    name = "medrag"

    def __init__(self, llm, retriever, k=32):
        self.llm = llm
        self.retriever = retriever
        self.k = k

    def __call__(self, q):
        # No keyword search, no re-ranking, no de-duplication, no expansion: the published setup.
        evidence = self.retriever.search(q.question, k=self.k, mode="dense", rerank=False,
                                         dedupe=False, expand=False)
        prompt = prompts.MEDRAG_USER.format(context=format_evidence(evidence),
                                            question=q.question, options=q.options_text())
        output = self.llm.chat(prompt, system=prompts.MEDRAG_SYSTEM, temperature=0.0)
        return {"answer": parse_choice(output, q.options),
                "trace": {"evidence": evidence_trace(evidence), "output": output}}


class RationaleRAG:
    """RAG²-style (Sohn et al., NAACL 2025): first write a short rationale, then search with it.
    The rationale is in textbook language, while the question is a patient's story, so it finds
    better passages. (RAG² also trains a small filter model; we skip training and use our
    hybrid search with re-ranking instead.)"""
    name = "rationale_rag"

    def __init__(self, llm, retriever, k=10):
        self.llm = llm
        self.retriever = retriever
        self.k = k

    def __call__(self, q):
        # The rationale is written from the question only - the options stay hidden (MIRAGE's rule).
        rationale = self.llm.chat(prompts.RATIONALE_USER.format(question=q.question),
                                  temperature=0.0, max_tokens=256)

        # Search with both the rationale and the question; each gets half of the budget first.
        evidence = self.retriever.search_many([rationale, q.question], budget=self.k, per_query=self.k // 2)

        prompt = prompts.MEDRAG_USER.format(context=format_evidence(evidence),
                                            question=q.question, options=q.options_text())
        output = self.llm.chat(prompt, system=prompts.MEDRAG_SYSTEM, temperature=0.0)
        return {"answer": parse_choice(output, q.options),
                "trace": {"rationale": rationale, "evidence": evidence_trace(evidence), "output": output}}


class IMedRAG:
    """i-MedRAG (Xiong et al., PSB 2025): over 3 rounds the model asks itself follow-up questions,
    each answered by plain RAG; the final answer sees all the questions and answers.

    Two small changes from the original code, both keeping the idea: the follow-up questions are
    asked for as JSON in the same call (the original needed an extra call to pull them out of free
    text), and the final answer uses MedRAG's prompt with the follow-ups written above the question.
    Like the original, the follow-ups are written while the options are visible.
    """
    name = "imedrag"

    def __init__(self, llm, retriever, rounds=3, n_queries=3, k=16):
        self.llm = llm
        self.retriever = retriever
        self.rounds = rounds
        self.n_queries = n_queries
        self.k = k

    def answer_follow_up(self, query):
        """One plain-RAG answer to one follow-up question."""
        evidence = self.retriever.search(query, k=self.k, mode="dense", rerank=False,
                                         dedupe=False, expand=False)
        prompt = prompts.SIMPLE_RAG_USER.format(context=format_evidence(evidence), question=query)
        return self.llm.chat(prompt, system=prompts.SIMPLE_RAG_SYSTEM, temperature=0.0, max_tokens=512)

    def history_text(self, history):
        blocks = []
        for query, answer in history:
            blocks.append("Query: " + query + "\nAnswer: " + answer)
        return "\n\n".join(blocks)

    def __call__(self, q):
        history = []            # [(follow-up question, its answer), ...]

        for round_number in range(self.rounds):
            prompt = prompts.IMEDRAG_ASK.format(context=self.history_text(history), question=q.question,
                                                options=q.options_text(), n_queries=self.n_queries)
            reply = self.llm.chat(prompt, system=prompts.IMEDRAG_SYSTEM, temperature=0.0)

            parsed = parse_json(reply) or {}
            new_queries = parsed.get("queries") or []
            if not isinstance(new_queries, list):      # sometimes it's just one string
                new_queries = [new_queries]
            for query in new_queries[:self.n_queries]:
                query = str(query)
                history.append((query, self.answer_follow_up(query)))

        final_question = self.history_text(history) + "\n\n" + q.question
        prompt = prompts.COT_USER.format(question=final_question, options=q.options_text())
        output = self.llm.chat(prompt, system=prompts.COT_SYSTEM, temperature=0.0)

        follow_ups = []
        for query, answer in history:
            follow_ups.append({"query": query, "answer": answer})
        return {"answer": parse_choice(output, q.options), "trace": {"follow_ups": follow_ups, "output": output}}


# name -> class; scripts/run_mirage.py picks methods by these names.
METHODS = {
    "cot": CoT,
    "cot_sc": SelfConsistency,
    "medrag": MedRAG,
    "rationale_rag": RationaleRAG,
    "imedrag": IMedRAG,
}

# These two never touch the library, so the runner doesn't have to load the big index for them.
RETRIEVAL_FREE = ["cot", "cot_sc"]
