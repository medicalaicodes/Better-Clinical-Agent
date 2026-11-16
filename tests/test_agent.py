"""
Tests for the clinical agent (agent.py), with the scripted fake LLM - no model server needed.
The agent only orders the blood-test tool here, so no image models are downloaded either.
"""

import json

from src.clinical_agent.agent import AttendingClinicalAgent
from src.clinical_agent.evaluation import BENCHMARK_CASES, STRESS_TEST_CASES
from src.clinical_agent.schemas import ClinicalReport, DispositionType

from tests.conftest import FakeLLM

INTAKE = json.dumps({
    "pertinent_positives": ["Fever", "Productive cough", "fever"],          # "fever" twice on purpose
    "pertinent_negatives": ["No orthopnea"],
    "differential": [
        {"condition": "Community-Acquired Pneumonia", "tier": "HIGH_SUSPICION", "clinical_rationale": "Fever and cough."},
        {"condition": "Acute Bronchitis", "tier": "moderate", "clinical_rationale": "Cough illness."},
    ],
})


def report_reply(diagnosis, disposition):
    return json.dumps({
        "primary_diagnosis": diagnosis,
        "disposition": disposition,
        "differential_diagnosis": [{"condition": diagnosis, "tier": "HIGH_SUSPICION", "clinical_rationale": "Evidence."}],
        "clinical_impression": "Summary of the evidence.",
        "recommendations": ["Start antibiotics", "Monitor oxygenation"],
    })


def labs_then_stop():
    """Orders the blood tests on the first turn, then says it has enough."""
    turns = []

    def reply(prompt):
        turns.append(prompt)
        if len(turns) == 1:
            return json.dumps({"selected_action": "laboratory_tests", "clinical_indication": "Check inflammation.",
                               "search_query": ""})
        return json.dumps({"selected_action": "STOP", "clinical_indication": "Enough evidence.", "search_query": ""})
    return reply


def test_agent_runs_the_whole_workup_through_the_graph():
    llm = FakeLLM([("patient intake evaluation", INTAKE),
                   ("Available Specialist Diagnostic Tools", labs_then_stop()),
                   ("finalizing the definitive", report_reply("Community-Acquired Pneumonia", "ADMIT_INPATIENT"))])
    agent = AttendingClinicalAgent(llm=llm)

    steps = []
    agent.on_event = lambda event, details: steps.append(event)
    report = agent.diagnose(BENCHMARK_CASES[0], max_turns=3)

    assert isinstance(report, ClinicalReport)
    assert report.primary_diagnosis == "Community-Acquired Pneumonia"
    assert report.disposition == DispositionType.ADMIT_INPATIENT
    assert [e.tool_name for e in report.evidence_trail] == ["clinical_history_synthesis", "laboratory_tests"]
    assert steps == ["intake", "choice", "tool_result", "choice", "report"]

    # the repeated "fever" was counted once
    intake = report.evidence_trail[0]
    assert intake.raw_output["pertinent_positives"] == ["Fever", "Productive cough"]


def test_turn_limit_stops_the_loop():
    always_labs = json.dumps({"selected_action": "laboratory_tests", "clinical_indication": "x", "search_query": ""})
    llm = FakeLLM([("patient intake evaluation", INTAKE), ("Available Specialist Diagnostic Tools", always_labs),
                   ("finalizing the definitive", report_reply("Pneumonia", "ADMIT_INPATIENT"))])
    report = AttendingClinicalAgent(llm=llm).diagnose(BENCHMARK_CASES[0], max_turns=1)
    assert len(report.evidence_trail) == 2          # intake + one tool, then the limit


def test_unclear_disposition_is_escalated_to_a_human():
    llm = FakeLLM([("patient intake evaluation", INTAKE),
                   ("Available Specialist Diagnostic Tools", json.dumps({"selected_action": "STOP"})),
                   ("finalizing the definitive", report_reply("Suspected aortic dissection", "transfer?"))])
    report = AttendingClinicalAgent(llm=llm).diagnose(STRESS_TEST_CASES[0])
    assert report.disposition == DispositionType.ESCALATE_TO_PHYSICIAN
