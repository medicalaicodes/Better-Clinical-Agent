import json

from src.clinical_agent.llm import parse_choice, parse_json
from src.clinical_agent.rag import METHODS, MCQ

from tests.conftest import FakeLLM, answer

QUESTION = MCQ(
    id="q1",
    question="A 34-year-old woman 2 weeks postpartum has sudden dyspnea, pleuritic chest pain, "
             "HR 118, SpO2 89%, clear lungs and a normal chest x-ray. Most appropriate next step?",
    options={"A": "D-dimer", "B": "CT pulmonary angiography", "C": "Echocardiography", "D": "Bronchoscopy"},
    answer="B",
)


def test_parsers_handle_messy_output():
    assert parse_json('Sure! ```json\n{"a": 1,}\n```') == {"a": 1}
    assert parse_choice('{"step_by_step_thinking": "x", "answer_choice": "C. Echo"}', "ABCD") == "C"
    assert parse_choice("so the answer is (B).", "ABCD") == "B"
    assert parse_choice("I am not sure", "ABCD") is None
    # Found in the first test run: the model copying the format hint "A/B/C/..." must NOT
    # be read as answer A.
    assert parse_choice('{"answer_choice": "A/B/C/..."}', "ABCD") is None
    assert parse_choice('"answer_choice": "A/B', "ABCD") is None


def test_cot_baseline():
    llm = FakeLLM([("Please think step-by-step", answer("B"))])
    assert METHODS["cot"](llm)(QUESTION)["answer"] == "B"


def test_self_consistency_takes_the_majority():
    llm = FakeLLM([("Please think step-by-step", [answer("A"), answer("B"), answer("B"), answer("C"), answer("B")])])
    result = METHODS["cot_sc"](llm)(QUESTION)
    assert result["answer"] == "B"
    assert result["trace"]["votes"]["B"] == 3


def test_rationale_rag_searches_with_its_rationale_without_options(toy_retriever):
    llm = FakeLLM([
        ("Write a short rationale", "Pulmonary embolism: CT pulmonary angiography confirms it."),
        ("Here are the relevant documents", answer("B")),
    ])
    result = METHODS["rationale_rag"](llm, toy_retriever, k=2)(QUESTION)
    assert result["answer"] == "B"
    rationale_prompt = [p for p in llm.prompts if "Write a short rationale" in p][0]
    assert "Bronchoscopy" not in rationale_prompt         # the options stay hidden


def test_imedrag_collects_follow_ups(toy_retriever):
    llm = FakeLLM([
        ("Generate 2 concise", json.dumps({"queries": ["D-dimer use", "CTPA indication"]})),
        ("Here are the relevant documents", "Short answer."),
        ("Please think step-by-step", answer("B")),
    ])
    result = METHODS["imedrag"](llm, toy_retriever, rounds=1, n_queries=2, k=2)(QUESTION)
    assert result["answer"] == "B"
    follow_up_questions = [f["query"] for f in result["trace"]["follow_ups"]]
    assert follow_up_questions == ["D-dimer use", "CTPA indication"]
