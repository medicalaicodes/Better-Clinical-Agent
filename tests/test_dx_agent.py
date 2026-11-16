"""
Tests for the diagnostic agent and its evidence tool, on CPU with a scripted fake LLM.

What is checked is the harness, not medical skill:
  - the answer options never reach the agent or the critic (only the mapper sees them);
  - the mapper never sees the case;
  - rule-outs without a real citation are rejected;
  - the critic's overlooked diagnosis is checked in the library like any other;
  - the agent <-> critic loop always ends within its caps;
  - evidence cards are whole, attributed and headed by their source.
"""

import json

import numpy as np

from src.clinical_agent.rag import METHODS, MCQ
from src.clinical_agent.rag.agent import DxAgent
from src.clinical_agent.retrieval.evidence_tool import EvidenceTool, complete_sentence, split_sentences
from src.clinical_agent.retrieval.search import word_set
from tests.conftest import FakeLLM

CASE = ("A 30-year-old woman develops sudden dyspnea and pleuritic chest pain after a long flight. "
        "Her D-dimer is raised and the chest radiograph is normal. "
        "Which of the following is the most likely diagnosis?")
OPTIONS = {"A": "Pneumothorax", "B": "Pulmonary embolism", "C": "Peripartum cardiomyopathy", "D": "Asthma"}


class OverlapReranker:
    """Stands in for the MedCPT cross-encoder: score = number of shared words."""

    def score(self, pairs):
        return np.array([len(word_set(q) & word_set(p)) for q, p in pairs], dtype=np.float32)


def frame_reply(prompt):
    return json.dumps({
        "findings": ["sudden dyspnea after a long flight", "raised D-dimer", "normal chest radiograph"],
        "asked": "the diagnosis",
        "hypotheses": [
            {"name": "Pulmonary embolism", "why": "dyspnea after a flight, raised D-dimer", "worst_fit": "none",
             "claims": [{"text": "D-dimer is raised in pulmonary embolism", "about": "tests"}]},
            {"name": "Pneumothorax", "why": "sudden dyspnea", "worst_fit": "normal chest radiograph",
             "claims": [{"text": "Pneumothorax causes absent breath sounds", "about": "presentation"}]},
        ],
        "leading": {"hypothesis": "Pulmonary embolism", "answer": "Pulmonary embolism"},
    })


def round_reply(prompt):
    if "Open issues from the critic" in prompt:
        # Second turn: answer the critic's issue and submit again.
        return json.dumps({"claim_status": [], "reasoning": "checked",
                           "responses": [{"issue": "I1", "reply": "a normal radiograph argues against it",
                                          "evidence": "F3"}],
                           "actions": [{"tool": "request_critique"}],
                           "leading": {"hypothesis": "Pulmonary embolism", "answer": "Pulmonary embolism"}})
    # First turn: one valid rule-out, one with a made-up citation, then ask for critique.
    return json.dumps({"claim_status": [{"claim": "C1", "status": "SUPPORTED", "card": "E1"}],
                       "reasoning": "PE explains everything",
                       "actions": [{"tool": "rule_out", "hypothesis": "Pneumothorax", "evidence": "E99",
                                    "reason": "made up"},
                                   {"tool": "rule_out", "hypothesis": "Pneumothorax", "evidence": "F3",
                                    "reason": "normal radiograph"},
                                   {"tool": "request_critique"}],
                       "leading": {"hypothesis": "Pulmonary embolism", "answer": "Pulmonary embolism"}})


def critic_reply(prompt):
    if "(none)" in prompt.split("Issues you raised before:")[1]:
        # First critique: one blocking issue and one overlooked diagnosis.
        return json.dumps({"strongest_alternative": {"hypothesis": "Pneumothorax", "case_for_it": "sudden onset"},
                           "issues": [{"type": "alternative", "detail": "pneumothorax not excluded by exam",
                                       "action": "check breath sounds", "changes_answer": True,
                                       "alternative_answer": "Pneumothorax"},
                                      {"type": "alternative", "detail": "vague doubt without an alternative",
                                       "action": "think", "changes_answer": True}],
                           "new_hypotheses": [{"name": "Peripartum cardiomyopathy", "why": "dyspnea",
                                               "claims": [{"text": "Peripartum cardiomyopathy causes heart failure near delivery",
                                                           "about": "presentation"}]}],
                           "restore": []})
    return json.dumps({"previous_issues": [{"issue": "I1", "resolved": True}], "issues": []})


def map_reply(prompt):
    return json.dumps({"matches": [{"answer": 1, "choice": "B"}]})


def make_llm():
    return FakeLLM([("Propose up to", frame_reply), ("turn(s) left", round_reply),
                    ("Review, in this order", critic_reply), ("Answers to match", map_reply)])


def test_agent_full_loop(toy_retriever):
    toy_retriever.reranker = OverlapReranker()
    llm = make_llm()
    agent = DxAgent(llm, toy_retriever, max_rounds=4, max_critiques=2)
    result = agent(MCQ("q1", CASE, OPTIONS, "B"))
    trace = result["trace"]

    assert result["answer"] == "B"
    assert trace["first"]["choice"] == "B" and not trace["mapping_fallback"]

    # The options never reach the agent or the critic; the mapper never sees the case.
    for prompt in llm.prompts:
        if "Answers to match" in prompt:
            assert "long flight" not in prompt
        else:
            assert "A. Pneumothorax" not in prompt and "D. Asthma" not in prompt

    # The made-up citation was rejected, the cited one accepted.
    events = [e["event"] for e in trace["events"]]
    assert "rule_out_rejected" in events and "rule_out" in events
    assert trace["hypotheses"]["Pneumothorax"]["status"] == "ruled_out"

    # The critic's overlooked diagnosis entered the differential and was checked in the library.
    added = trace["hypotheses"]["Peripartum cardiomyopathy"]
    assert added["origin"] == "critic"
    assert added["claims"][0]["cards"]                    # looked up, found evidence

    # Two critiques, then the loop ended: the second found nothing new.
    assert trace["critiques"] == 2 and trace["open_issues"] == []

    # A "changes the answer" issue that names no alternative answer does not block.
    vague = [i for i in trace["issues"].values() if i["detail"].startswith("vague")][0]
    assert vague["changes_answer"] is False


def test_loop_always_ends_when_the_critic_never_agrees(toy_retriever):
    def stubborn_critic(prompt):
        number = prompt.count("colleague replied") + 1
        return json.dumps({"issues": [{"type": "alternative", "detail": f"new doubt number {number}",
                                       "action": "think again", "changes_answer": True,
                                       "alternative_answer": f"Something else {number}"}]})

    def flip_flopping_agent(prompt):
        answer = "Pneumothorax" if prompt.count("colleague replied") % 2 else "Pulmonary embolism"
        return json.dumps({"actions": [{"tool": "request_critique"}],
                           "leading": {"hypothesis": answer, "answer": answer}})

    toy_retriever.reranker = OverlapReranker()
    llm = FakeLLM([("Propose up to", frame_reply), ("turn(s) left", flip_flopping_agent),
                   ("Review, in this order", stubborn_critic), ("Answers to match", map_reply)])
    agent = DxAgent(llm, toy_retriever, max_rounds=3, max_critiques=2)
    trace = agent(MCQ("q2", CASE, OPTIONS, "B"))["trace"]

    assert trace["turns"] <= 3 and trace["critiques"] <= 2
    assert trace["open_issues"]                            # unresolved doubt is recorded, not looped on


def test_frame_only_mode_uses_two_calls_and_no_retriever():
    llm = make_llm()
    agent = DxAgent(llm, retriever=None, max_rounds=0)
    result = agent(MCQ("q3", CASE, OPTIONS, "B"))
    assert result["answer"] == "B"
    assert len(llm.prompts) == 2                           # frame + map


def test_agent_is_registered_as_a_method():
    assert METHODS["dx_agent"] is DxAgent


def test_evidence_cards_are_attributed_and_whole(toy_retriever):
    toy_retriever.reranker = OverlapReranker()
    tool = EvidenceTool(toy_retriever, cards_per_claim=2, min_score=1.0)

    assert tool.find_article("Pulmonary embolism") == "pulmonary embolism"
    assert tool.find_article("Crigler-Najjar syndrome type II") is None

    cards = tool.consult([{"hypothesis": "Pulmonary embolism", "claim": "D-dimer excludes pulmonary embolism",
                           "about": "tests"},
                          {"hypothesis": "Asthma", "claim": "wheeze responds to bronchodilators", "about": "other"}])
    first = cards[0][0]
    assert first["source"] == "StatPearls: Pulmonary Embolism, Evaluation"
    assert first["text"].endswith(".")                    # a whole sentence, never cut
    assert cards[1] == []                                 # nothing relevant -> NOT FOUND, no filler


def test_sentence_completion_keeps_subject_and_lists():
    sentences = split_sentences("Gilbert syndrome is common. It is caused by reduced UGT1A1 activity. "
                                "Triggers include:\nFasting.\nIllness.")
    assert complete_sentence(sentences, 1) == [0, 1]           # "It" needs the sentence before
    assert complete_sentence(sentences, 2) == [2, 3, 4]        # "include:" needs its list


def test_select_mode_reveals_options_only_at_the_end(toy_retriever):
    toy_retriever.reranker = OverlapReranker()

    def select_reply(prompt):
        return json.dumps({"reasoning": "matches", "answer_choice": "B"})

    llm = FakeLLM([("Propose up to", frame_reply), ("turn(s) left", round_reply),
                   ("Review, in this order", critic_reply), ("Answers to match", map_reply),
                   ("Your work-up", select_reply)])
    result = DxAgent(llm, toy_retriever, final="select")(MCQ("q4", CASE, OPTIONS, "B"))
    assert result["answer"] == "B" and result["trace"]["final_step"] == "select"

    # Only the mapper and the final selection ever see the options list.
    seen = [p for p in llm.prompts if "A. Pneumothorax" in p]
    assert len(seen) == 2 and "Your work-up" in seen[-1]


def test_invented_claim_ids_and_repeated_claims_are_ignored(toy_retriever):
    toy_retriever.reranker = OverlapReranker()

    def chatty_agent(prompt):
        # Judges a real claim, invents C99, and asks for a claim PE already has.
        return json.dumps({"claim_status": [{"claim": "C1", "status": "SUPPORTED", "card": "E1"},
                                            {"claim": "C99", "status": "SUPPORTED", "card": "none"}],
                           "actions": [{"tool": "consult_reference", "hypothesis": "Pulmonary embolism",
                                        "claim": "D-dimer is raised in pulmonary embolism", "about": "tests"},
                                       {"tool": "request_critique"}],
                           "leading": {"hypothesis": "Pulmonary embolism", "answer": "Pulmonary embolism"}})

    llm = FakeLLM([("Propose up to", frame_reply), ("turn(s) left", chatty_agent),
                   ("Review, in this order", critic_reply), ("Answers to match", map_reply)])
    trace = DxAgent(llm, toy_retriever, max_rounds=1)(MCQ("q5", CASE, OPTIONS, "B"))["trace"]

    pe_claims = trace["hypotheses"]["Pulmonary embolism"]["claims"]
    assert len(pe_claims) == 1                          # the repeat was not added again
    assert pe_claims[0]["status"] == "SUPPORTED"
    all_claims = [c for h in trace["hypotheses"].values() for c in h["claims"]]
    assert all(c["text"] != "C99" for c in all_claims)
