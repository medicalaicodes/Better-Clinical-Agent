"""
Shared test fixtures: a tiny in-memory corpus with a BM25-only retriever, and a scripted
fake LLM. Tests run on CPU in seconds, with no model downloads and no LLM server.
"""

import json
import threading

import pytest

from src.clinical_agent.retrieval import CorpusStore, Retriever, Snippet
from src.clinical_agent.retrieval.indexes import BM25Index

TOY_SNIPPETS = [
    Snippet("PE_0", "Pulmonary Embolism -- Evaluation",
            "In patients with high pretest probability of pulmonary embolism, proceed directly to CT pulmonary angiography.", "statpearls"),
    Snippet("PE_1", "Pulmonary Embolism -- Evaluation",
            "D-dimer is useful to exclude pulmonary embolism when pretest probability is low or intermediate.", "statpearls"),
    Snippet("PE_2", "Pulmonary Embolism -- Evaluation",
            "The chest radiograph is often normal in pulmonary embolism.", "statpearls"),
    Snippet("PPCM_0", "Peripartum Cardiomyopathy -- History",
            "Peripartum cardiomyopathy presents with heart failure symptoms and pulmonary edema near delivery.", "statpearls"),
    Snippet("PTX_0", "Pneumothorax -- Exam",
            "Pneumothorax causes absent breath sounds and hyperresonance on the affected side.", "statpearls"),
    Snippet("Pharm_0", "Pharmacology_Katzung",
            "Nitrofurantoin is used for uncomplicated cystitis in pregnancy.", "textbooks"),
]


@pytest.fixture
def toy_retriever():
    corpus = CorpusStore(TOY_SNIPPETS)
    bm25 = BM25Index().build(list(corpus.texts()))
    return Retriever(corpus, bm25=bm25)


class FakeLLM:
    """Answers by matching a keyword in the prompt; records every prompt it was sent."""

    def __init__(self, rules):
        self.rules = rules              # list of (substring, reply or callable)
        self.prompts = []
        self._local = threading.local()

    def reset_usage(self):
        self._local.calls = 0

    def usage(self):
        return {"calls": getattr(self._local, "calls", 0), "prompt_tokens": 0, "completion_tokens": 0}

    def chat(self, prompt, system=None, n=1, **kwargs):
        self.prompts.append(prompt)
        self._local.calls = getattr(self._local, "calls", 0) + 1
        for key, reply in self.rules:
            if key in prompt:
                replies = reply(prompt) if callable(reply) else reply
                break
        else:
            replies = "{}"
        if isinstance(replies, list):
            return replies[:n] if n > 1 else replies[0]
        return [replies] * n if n > 1 else replies


def answer(letter):
    return json.dumps({"step_by_step_thinking": "...", "answer_choice": letter})
