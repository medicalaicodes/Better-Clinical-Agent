import json

from src.clinical_agent.llm import parse_choice, parse_json
from src.clinical_agent.rag import METHODS, MCQ, DDxRAG

from tests.conftest import FakeLLM, answer

QUESTION = MCQ(
    id="q1",
    question="A 34-year-old woman 2 weeks postpartum has sudden dyspnea, pleuritic chest pain, "
             "HR 118, SpO2 89%, clear lungs and a normal chest x-ray. Most appropriate next step?",
    options={"A": "D-dimer", "B": "CT pulmonary angiography", "C": "Echocardiography", "D": "Bronchoscopy"},
    answer="B",
)

PLAN = json.dumps({
    "findings": ["postpartum", "sudden dyspnea", "hypoxemia", "normal chest x-ray"],
    "hypotheses": ["pulmonary embolism", "peripartum cardiomyopathy"],
    "queries": [
        {"intent": "SUPPORT", "about": "pulmonary embolism", "text": "pulmonary embolism CT pulmonary angiography"},
        {"intent": "DISCRIMINATE", "about": "pulmonary embolism", "text": "D-dimer pretest probability"},
        {"intent": "REFUTE", "about": "pulmonary embolism", "text": "chest radiograph normal pulmonary embolism"},
    ],
})

VERIFY = json.dumps({"hypotheses": [
    {"name": "pulmonary embolism", "for": [{"point": "high pretest -> CTPA", "evidence": "E1"},
                                           {"point": "made up", "evidence": "E99"}],
     "against": [], "verdict": "supported"},
]})


def make_llm(votes):
    return FakeLLM([
        ("Plan the literature check", PLAN),
        ("fact-checker", VERIFY),               # never matched: system prompts are not in `prompt`
        ("For EACH hypothesis", VERIFY),
        ("Reference evidence", answer("B")),     # decide step
        ("Please think step-by-step", votes),   # closed-book votes
    ])


def test_parsers_handle_messy_output():
    assert parse_json('Sure! ```json\n{"a": 1,}\n```') == {"a": 1}
    assert parse_choice('{"step_by_step_thinking": "x", "answer_choice": "C. Echo"}', "ABCD") == "C"
    assert parse_choice("so the answer is (B).", "ABCD") == "B"
    assert parse_choice("I am not sure", "ABCD") is None
    # regression (found in the CPU smoke test): the model echoing the format hint
    # "A/B/C/..." must NOT be read as answer A
    assert parse_choice('{"answer_choice": "A/B/C/..."}', "ABCD") is None
    assert parse_choice('"answer_choice": "A/B', "ABCD") is None


def test_placeholder_queries_from_the_plan_are_dropped(toy_retriever):
    echoed_plan = json.dumps({"findings": ["..."], "hypotheses": ["<most likely answer>"],
                              "queries": [{"intent": "SUPPORT", "about": "...", "text": "..."}]})
    llm = FakeLLM([("Plan the literature check", echoed_plan)])
    plan = DDxRAG(llm, toy_retriever).plan(QUESTION)
    assert plan == {"findings": [], "hypotheses": [], "queries": []}


def test_cot_baseline():
    llm = FakeLLM([("Please think step-by-step", answer("B"))])
    assert METHODS["cot"](llm)(QUESTION)["answer"] == "B"


def test_ddx_rag_full_path_when_votes_disagree(toy_retriever):
    llm = make_llm([answer("A"), answer("A"), answer("B"), answer("B"), answer("A")])
    result = DDxRAG(llm, toy_retriever, n_votes=5, gate_threshold=0.8)(QUESTION)
    trace = result["trace"]

    assert result["answer"] == "B"
    assert trace["path"] == "full"
    assert trace["agreement"] == 0.6
    assert trace["changed_from_majority"] is True
    assert {q["intent"] for q in trace["queries"]} >= {"SUPPORT", "DISCRIMINATE", "REFUTE", "QUESTION"}
    # the point citing a non-existent E99 was dropped by the grounding check
    assert trace["verification"]["uncited_points_dropped"] == 1
    # the plan prompt never saw the answer options (question-only retrieval)
    plan_prompt = next(p for p in llm.prompts if "Plan the literature check" in p)
    assert "CT pulmonary angiography" not in plan_prompt and "Bronchoscopy" not in plan_prompt


def test_ddx_rag_light_path_when_votes_agree(toy_retriever):
    llm = make_llm([answer("B")] * 5)
    result = DDxRAG(llm, toy_retriever, n_votes=5, gate_threshold=0.8)(QUESTION)
    assert result["trace"]["path"] == "light"
    assert result["trace"]["verification"] is None
    assert not any("For EACH hypothesis" in p for p in llm.prompts)   # no verify call


def test_ddx_rag_ablation_switches(toy_retriever):
    llm = make_llm([answer("A"), answer("B"), answer("C"), answer("A"), answer("B")])
    method = DDxRAG(llm, toy_retriever, intents="SUPPORT", verify=False, show_votes=False)
    trace = method(QUESTION)["trace"]
    assert {q["intent"] for q in trace["queries"]} == {"SUPPORT", "QUESTION"}
    assert trace["verification"] is None
    decide_prompt = next(p for p in llm.prompts if "Reference evidence" in p)
    assert "WITHOUT references" not in decide_prompt


def test_imedrag_collects_follow_ups(toy_retriever):
    llm = FakeLLM([
        ("Generate 2 concise", json.dumps({"queries": ["D-dimer use", "CTPA indication"]})),
        ("Here are the relevant documents", "Short answer."),
        ("Please think step-by-step", answer("B")),
    ])
    result = METHODS["imedrag"](llm, toy_retriever, rounds=1, n_queries=2, k=2)(QUESTION)
    assert result["answer"] == "B"
    assert [f["query"] for f in result["trace"]["follow_ups"]] == ["D-dimer use", "CTPA indication"]
