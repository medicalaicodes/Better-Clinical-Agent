from src.clinical_agent.evaluation.mirage import comparison_table, run_method
from src.clinical_agent.evaluation.stats import mcnemar_exact, wilson_interval
from src.clinical_agent.rag import MCQ

from tests.conftest import FakeLLM


def test_wilson_interval_contains_the_estimate():
    low, high = wilson_interval(64, 100)
    assert low < 0.64 < high
    assert round(low, 2) == 0.54 and round(high, 2) == 0.73


def test_mcnemar_only_counts_discordant_pairs():
    a = [True] * 50 + [False] * 10 + [True] * 2
    b = [True] * 50 + [True] * 10 + [False] * 2
    result = mcnemar_exact(a, b)
    assert (result["b01"], result["b10"]) == (10, 2)
    assert result["p_value"] < 0.05


def test_runner_resumes_and_reports(tmp_path):
    questions = [MCQ(str(i), f"q{i}", {"A": "a", "B": "b"}, answer="A") for i in range(6)]
    llm = FakeLLM([])
    calls = []

    def always_a(q):
        calls.append(q.id)
        return {"answer": "A" if int(q.id) % 2 == 0 else "B", "trace": {}}

    out = tmp_path / "m.jsonl"
    run_method(always_a, questions[:3], str(out), llm, workers=2)
    results = run_method(always_a, questions, str(out), llm, workers=2)
    assert sorted(calls) == ["0", "1", "2", "3", "4", "5"]      # first 3 not re-run
    assert len(results) == 6

    table = comparison_table({"m": results, "m2": results}, baseline="m")
    assert "| m2 | 6 | 50.0 |" in table
