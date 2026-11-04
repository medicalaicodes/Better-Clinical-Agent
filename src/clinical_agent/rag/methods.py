"""
Medical question-answering methods: five baselines from the literature, and DDx-RAG (ours).

Every method is a small class used the same way:

    method = DDxRAG(llm, retriever, **settings)
    result = method(question)          # -> {"answer": "B", "trace": {...}}

`answer` is an option letter, or None if the model never committed to one.
`trace` keeps every intermediate step, so any prediction can be audited later.

    name           retrieval                                          LLM calls per question
    -------------  -------------------------------------------------  ----------------------
    cot            none (closed book)                                 1
    cot_sc         none, majority vote of n samples                   1 (n samples)
    medrag         question -> MedCPT top-32                          1
    rationale_rag  LLM rationale -> hybrid + rerank   (RAG²-style)    2
    imedrag        iterative LLM follow-up queries                    ~2 + rounds x (1 + queries)
    ddx_rag        support / discriminate / refute queries,           3 (light) or 4 (full)
                   cited fact-check, adjudication against the votes
"""

from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional

from ..llm import LLM, is_placeholder, parse_choice, parse_json
from ..retrieval import Evidence, Query, Retriever
from . import prompts


@dataclass
class MCQ:
    """One multiple-choice question."""
    id: str
    question: str
    options: Dict[str, str]     # {"A": "...", "B": "...", ...}
    answer: str = ""            # gold letter (methods never look at it)
    dataset: str = ""

    def options_text(self) -> str:
        lines = []
        for letter in sorted(self.options):
            lines.append(f"{letter}. {self.options[letter]}")
        return "\n".join(lines)


def format_evidence(evidence: List[Evidence], with_ids: bool = True, max_chars: int = 1500) -> str:
    """Turns evidence into the numbered block the prompts show the model:
        [E1] (statpearls: Pulmonary Embolism -- Evaluation) text...
    Baselines use MedRAG's "Document [1]" style instead of ids."""
    if not evidence:
        return "(no evidence retrieved)"

    lines = []
    for number, item in enumerate(evidence, start=1):
        label = f"[E{number}] " if with_ids else f"Document [{number}] "
        lines.append(f"{label}({item.source}: {item.title}) {item.text[:max_chars]}")
    return "\n".join(lines)


def evidence_trace(evidence: List[Evidence]) -> List[dict]:
    """The compact form of the evidence saved in result files."""
    saved = []
    for item in evidence:
        saved.append({"id": item.snippet.id, "score": round(item.score, 3), "intents": item.intents})
    return saved


# ========================================================================================
# Baselines
# ========================================================================================

class CoT:
    """Closed-book chain of thought, with MedRAG's exact prompt."""
    name = "cot"

    def __init__(self, llm: LLM, retriever: Optional[Retriever] = None):
        self.llm = llm

    def __call__(self, q: MCQ) -> dict:
        prompt = prompts.COT_USER.format(question=q.question, options=q.options_text())
        output = self.llm.chat(prompt, system=prompts.COT_SYSTEM, temperature=0.0)
        return {"answer": parse_choice(output, q.options), "trace": {"output": output}}


def sample_votes(llm: LLM, q: MCQ, n: int, temperature: float) -> Counter:
    """Asks the closed-book CoT question n times (in one request) and counts the letters.
    Example result: Counter({"B": 3, "A": 2}). Unparseable answers are counted as None."""
    prompt = prompts.COT_USER.format(question=q.question, options=q.options_text())
    outputs = llm.chat(prompt, system=prompts.COT_SYSTEM, temperature=temperature, n=n, seed=0)
    if not isinstance(outputs, list):
        outputs = [outputs]

    votes = Counter()
    for text in outputs:
        votes[parse_choice(text, q.options)] += 1
    return votes


def majority(votes: Counter) -> Optional[str]:
    """Most common valid letter. Ties go to the later letter (deterministic, arbitrary)."""
    best = None
    for letter, count in votes.items():
        if letter is None:
            continue
        if best is None or (count, letter) > (votes[best], best):
            best = letter
    return best


class SelfConsistency:
    """Closed-book majority vote over n sampled answers (Wang et al., ICLR 2023)."""
    name = "cot_sc"

    def __init__(self, llm: LLM, retriever: Optional[Retriever] = None, n: int = 5,
                 temperature: float = 0.7):
        self.llm = llm
        self.n = n
        self.temperature = temperature

    def __call__(self, q: MCQ) -> dict:
        votes = sample_votes(self.llm, q, self.n, self.temperature)
        return {"answer": majority(votes), "trace": {"votes": dict(votes)}}


class MedRAG:
    """Vanilla RAG as in MIRAGE: the whole question is the query, MedCPT dense retrieval,
    top-32 snippets pasted into MedRAG's prompt."""
    name = "medrag"

    def __init__(self, llm: LLM, retriever: Retriever, k: int = 32):
        self.llm = llm
        self.retriever = retriever
        self.k = k

    def __call__(self, q: MCQ) -> dict:
        # No hybrid, no rerank, no dedupe, no expansion: the plain published setup.
        evidence = self.retriever.search(q.question, k=self.k, mode="dense", rerank=False,
                                         dedupe=False, expand=False)
        prompt = prompts.MEDRAG_USER.format(context=format_evidence(evidence, with_ids=False),
                                            question=q.question, options=q.options_text())
        output = self.llm.chat(prompt, system=prompts.MEDRAG_SYSTEM, temperature=0.0)
        return {"answer": parse_choice(output, q.options),
                "trace": {"evidence": evidence_trace(evidence), "output": output}}


class RationaleRAG:
    """RAG²-style (Sohn et al., NAACL 2025): first write a short rationale, then use THAT as
    the search query, because it is written in textbook language while the question is a
    patient story. RAG² also trains a small filter model; we skip that (no training) and
    use hybrid search plus the cross-encoder instead."""
    name = "rationale_rag"

    def __init__(self, llm: LLM, retriever: Retriever, k: int = 10):
        self.llm = llm
        self.retriever = retriever
        self.k = k

    def __call__(self, q: MCQ) -> dict:
        # The rationale is written from the question only (no options): MIRAGE's rule.
        rationale = self.llm.chat(prompts.RATIONALE_USER.format(question=q.question),
                                  temperature=0.0, max_tokens=256)

        queries = [Query(rationale, "RATIONALE"), Query(q.question, "QUESTION")]
        evidence = self.retriever.search_many(queries, budget=self.k, per_query=self.k // 2)

        prompt = prompts.MEDRAG_USER.format(context=format_evidence(evidence, with_ids=False),
                                            question=q.question, options=q.options_text())
        output = self.llm.chat(prompt, system=prompts.MEDRAG_SYSTEM, temperature=0.0)
        return {"answer": parse_choice(output, q.options),
                "trace": {"rationale": rationale, "evidence": evidence_trace(evidence), "output": output}}


class IMedRAG:
    """i-MedRAG (Xiong et al., PSB 2025): the LLM asks itself follow-up questions over
    several rounds; each follow-up is answered by vanilla RAG; the final answer sees all
    the follow-up questions and answers.

    Differences from the original code (both keep the idea, and both are cheaper):
      - follow-up queries are requested as JSON in the same call (the original makes a
        second LLM call just to parse them out of free text);
      - the final answer uses MedRAG's CoT prompt with the follow-ups prepended.
    Like the original, follow-ups are written while the OPTIONS ARE VISIBLE, so this
    baseline sees more at retrieval time than DDx-RAG does (it is not MIRAGE "QOR"-compliant).
    """
    name = "imedrag"

    def __init__(self, llm: LLM, retriever: Retriever, rounds: int = 3, n_queries: int = 3, k: int = 16):
        self.llm = llm
        self.retriever = retriever
        self.rounds = rounds
        self.n_queries = n_queries
        self.k = k

    def answer_follow_up(self, query: str) -> str:
        """One vanilla-RAG answer to one follow-up question."""
        evidence = self.retriever.search(query, k=self.k, mode="dense", rerank=False,
                                         dedupe=False, expand=False)
        prompt = prompts.SIMPLE_RAG_USER.format(context=format_evidence(evidence, with_ids=False),
                                                question=query)
        return self.llm.chat(prompt, system=prompts.SIMPLE_RAG_SYSTEM, temperature=0.0, max_tokens=512)

    @staticmethod
    def history_text(history: List[tuple]) -> str:
        blocks = []
        for query, answer in history:
            blocks.append(f"Query: {query}\nAnswer: {answer}")
        return "\n\n".join(blocks)

    def __call__(self, q: MCQ) -> dict:
        history = []          # [(follow-up query, its answer), ...]

        for _round in range(self.rounds):
            prompt = prompts.IMEDRAG_ASK.format(context=self.history_text(history), question=q.question,
                                                options=q.options_text(), n_queries=self.n_queries)
            reply = self.llm.chat(prompt, system=prompts.IMEDRAG_SYSTEM, temperature=0.0)

            parsed = parse_json(reply) or {}
            new_queries = parsed.get("queries") or []
            if not isinstance(new_queries, list):      # e.g. the model returned one string
                new_queries = [new_queries]
            for query in new_queries[: self.n_queries]:
                query = str(query)
                history.append((query, self.answer_follow_up(query)))

        final_question = f"{self.history_text(history)}\n\n{q.question}"
        prompt = prompts.COT_USER.format(question=final_question, options=q.options_text())
        output = self.llm.chat(prompt, system=prompts.COT_SYSTEM, temperature=0.0)

        follow_ups = [{"query": query, "answer": answer} for query, answer in history]
        return {"answer": parse_choice(output, q.options),
                "trace": {"follow_ups": follow_ups, "output": output}}


# ========================================================================================
# DDx-RAG (ours)
# ========================================================================================

class DDxRAG:
    """Hypothesis-driven, falsification-aware RAG with uncertainty-gated verification.

    0. VOTE     answer n times WITHOUT references -> how much the model agrees with itself.
    1. PLAN     from the question ONLY (options hidden): findings, 2-4 hypotheses, and short
                search queries tagged SUPPORT / DISCRIMINATE / REFUTE.
    2. GATE     high agreement -> light path: only check the leading hypothesis.
                low agreement  -> full path: check the whole differential.
    3. RETRIEVE hybrid search + rerank, with a quota per query, so evidence AGAINST the
                leading hypothesis always makes it into the prompt.
    4. VERIFY   (full path) fact-check every hypothesis; each point must cite an evidence id,
                points citing nothing real are dropped.
    5. DECIDE   options visible; the model sees the evidence, the fact-check and its own
                closed-book tallies, and may only overturn the majority by citing evidence.

    Every switch in __init__ exists so that its contribution can be measured by an ablation.
    """
    name = "ddx_rag"

    def __init__(self, llm: LLM, retriever: Retriever,
                 n_votes: int = 5, vote_temperature: float = 0.7,
                 gate_threshold: float = 0.8,
                 intents: str = "SUPPORT,DISCRIMINATE,REFUTE",
                 verify: bool = True, show_votes: bool = True,
                 budget: int = 10, light_budget: int = 5, per_query: int = 2,
                 min_score: Optional[float] = None, expand: bool = True, mode: str = "hybrid"):
        self.llm = llm
        self.retriever = retriever

        self.n_votes = n_votes                  # closed-book samples for the agreement signal
        self.vote_temperature = vote_temperature
        self.gate_threshold = gate_threshold    # agreement >= this -> light path (1.01 = never)

        # Which query intents are allowed ("SUPPORT,DISCRIMINATE,REFUTE" by default).
        self.intents = set()
        for intent in intents.split(","):
            if intent.strip():
                self.intents.add(intent.strip().upper())

        self.verify = verify                    # run the fact-check step on the full path
        self.show_votes = show_votes            # show closed-book tallies to the decider

        self.budget = budget                    # evidence snippets on the full path
        self.light_budget = light_budget        # evidence snippets on the light path
        self.per_query = per_query              # guaranteed snippets per query
        self.min_score = min_score              # cross-encoder cut-off (None = no cut-off)
        self.expand = expand                    # add neighbouring paragraphs
        self.mode = mode                        # "hybrid" | "bm25" | "dense"

    # ------------------------------------------------------------------------------------
    # Step 1: PLAN (question only)
    # ------------------------------------------------------------------------------------
    def plan(self, q: MCQ) -> dict:
        reply = self.llm.chat(prompts.PLAN_USER.format(question=q.question), system=prompts.PLAN_SYSTEM,
                              temperature=0.0, max_tokens=600)
        parsed = parse_json(reply) or {}

        # Small models sometimes copy the prompt's example ("...") instead of filling it in;
        # is_placeholder() filters those out everywhere below.
        findings = []
        for item in parsed.get("findings", []):
            if not is_placeholder(item):
                findings.append(str(item))

        hypotheses = []
        for item in parsed.get("hypotheses", []):
            if not is_placeholder(item):
                hypotheses.append(str(item))

        queries = []
        for item in parsed.get("queries", [])[:6]:
            if not isinstance(item, dict):
                continue
            text = str(item.get("text", "")).strip()
            if is_placeholder(text):
                continue
            text = " ".join(text.split()[:20])      # retrievers work best with short queries
            intent = str(item.get("intent", "SUPPORT")).upper()
            about = str(item.get("about", ""))
            queries.append(Query(text, intent, about))

        return {"findings": findings[:8], "hypotheses": hypotheses[:4], "queries": queries}

    # ------------------------------------------------------------------------------------
    # Step 2: which queries to run (depends on the gate)
    # ------------------------------------------------------------------------------------
    def choose_queries(self, q: MCQ, plan: dict, light: bool) -> List[Query]:
        queries = [query for query in plan["queries"] if query.intent in self.intents]

        if light and plan["hypotheses"]:
            # Light path: the model is already fairly sure, so we only check its leading
            # hypothesis - what supports it and what argues against it.
            lead = plan["hypotheses"][0].lower()
            about_lead = []
            for query in queries:
                if query.intent not in ("SUPPORT", "REFUTE"):
                    continue
                about = query.about.lower()
                if not about or about in lead or lead in about:
                    about_lead.append(query)
            queries = about_lead or queries[:2]

        # The question itself is always added as a query, so retrieval still finds
        # something useful even when the plan was poor or failed to parse.
        queries.append(Query(q.question, "QUESTION"))
        return queries

    # ------------------------------------------------------------------------------------
    # Step 4: VERIFY (cited fact-check of each hypothesis)
    # ------------------------------------------------------------------------------------
    def verify_hypotheses(self, q: MCQ, plan: dict, evidence: List[Evidence]) -> dict:
        numbered = []
        for number, hypothesis in enumerate(plan["hypotheses"], start=1):
            numbered.append(f"{number}. {hypothesis}")

        prompt = prompts.VERIFY_USER.format(
            question=q.question,
            findings="; ".join(plan["findings"]) or "(see question)",
            hypotheses="\n".join(numbered),
            evidence=format_evidence(evidence),
        )
        reply = self.llm.chat(prompt, system=prompts.VERIFY_SYSTEM, temperature=0.0)
        table = parse_json(reply) or {}

        # Grounding check: a point only counts if it cites an evidence id that exists
        # (E1..En). Points citing "E99" or nothing at all are hallucinated: drop them.
        valid_ids = set()
        for number in range(1, len(evidence) + 1):
            valid_ids.add(f"E{number}")

        rows = []
        dropped = 0
        raw_rows = table.get("hypotheses")
        if not isinstance(raw_rows, list):
            raw_rows = []

        for raw in raw_rows:
            if not isinstance(raw, dict):
                continue
            row = {"name": str(raw.get("name", "")), "for": [], "against": [],
                   "verdict": str(raw.get("verdict", "no evidence"))}
            for side in ("for", "against"):
                for point in raw.get(side) or []:
                    cited = isinstance(point, dict) and str(point.get("evidence", "")).strip() in valid_ids
                    has_content = isinstance(point, dict) and not is_placeholder(point.get("point", ""))
                    if cited and has_content:
                        row[side].append(point)
                    else:
                        dropped += 1
            rows.append(row)

        return {"rows": rows, "uncited_points_dropped": dropped}

    @staticmethod
    def render_verification(verification: Optional[dict]) -> str:
        """The fact-check table as plain text for the DECIDE prompt."""
        if not verification or not verification["rows"]:
            return ""

        lines = ["", "Fact-check of the most likely answers (from the evidence, before seeing the options):"]
        for row in verification["rows"]:
            pro = "; ".join(f"{p.get('point', '')} [{p['evidence']}]" for p in row["for"]) or "none"
            con = "; ".join(f"{p.get('point', '')} [{p['evidence']}]" for p in row["against"]) or "none"
            lines.append(f"- {row['name']} -> {row['verdict']}. For: {pro}. Against: {con}.")
        return "\n".join(lines) + "\n"

    # ------------------------------------------------------------------------------------
    # Step 5: DECIDE (options visible)
    # ------------------------------------------------------------------------------------
    def decide(self, q: MCQ, evidence: List[Evidence], verification: Optional[dict], votes: Counter) -> str:
        votes_text = ""
        vote_rule = ""
        if self.show_votes:
            tally = []
            for letter, count in votes.most_common():
                if letter:
                    tally.append(f"{letter}: {count}")
            votes_text = (f"\nIndependent answers WITHOUT references ({sum(votes.values())} tries): "
                          f"{', '.join(tally)}\n")
            vote_rule = prompts.VOTE_RULE

        prompt = prompts.DECIDE_USER.format(
            question=q.question,
            options=q.options_text(),
            evidence=format_evidence(evidence),
            verification=self.render_verification(verification),
            votes=votes_text,
            vote_rule=vote_rule,
        )
        return self.llm.chat(prompt, system=prompts.DECIDE_SYSTEM, temperature=0.0)

    # ------------------------------------------------------------------------------------
    # The whole pipeline
    # ------------------------------------------------------------------------------------
    def __call__(self, q: MCQ) -> dict:
        # 0. VOTE: e.g. B, B, A, B, A -> majority B, agreement 3/5 = 0.6
        votes = sample_votes(self.llm, q, self.n_votes, self.vote_temperature)
        top = majority(votes)
        total = max(1, sum(votes.values()))
        agreement = votes[top] / total if top else 0.0

        # 1. PLAN (no options)
        plan = self.plan(q)

        # 2. GATE
        light = agreement >= self.gate_threshold
        queries = self.choose_queries(q, plan, light)

        # 3. RETRIEVE
        evidence = self.retriever.search_many(
            queries,
            budget=self.light_budget if light else self.budget,
            per_query=self.per_query,
            mode=self.mode,
            min_score=self.min_score,
            expand=self.expand,
        )

        # 4. VERIFY (full path only)
        verification = None
        if self.verify and not light and plan["hypotheses"]:
            verification = self.verify_hypotheses(q, plan, evidence)

        # 5. DECIDE
        output = self.decide(q, evidence, verification, votes)
        answer = parse_choice(output, q.options)
        if answer is None:
            answer = top          # the model never committed: fall back to the closed-book vote

        return {
            "answer": answer,
            "trace": {
                "votes": {str(letter): count for letter, count in votes.items()},
                "agreement": round(agreement, 2),
                "path": "light" if light else "full",
                "findings": plan["findings"],
                "hypotheses": plan["hypotheses"],
                "queries": [{"intent": x.intent, "about": x.about, "text": x.text} for x in queries],
                "evidence": evidence_trace(evidence),
                "verification": verification,
                "output": output,
                "changed_from_majority": answer != top,
            },
        }


# name -> class, used by scripts/run_mirage.py
METHODS = {}
for _method in (CoT, SelfConsistency, MedRAG, RationaleRAG, IMedRAG, DDxRAG):
    METHODS[_method.name] = _method

RETRIEVAL_FREE = {"cot", "cot_sc"}     # these never need the (large) retrieval index loaded
