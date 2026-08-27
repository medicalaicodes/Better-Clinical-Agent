"""
Clinical AI Agent: Unit Tests for Attending Agent & Graph Workflow
"""

import json
import pytest
from unittest.mock import patch
from src.clinical_agent import (
    AttendingClinicalAgent,
    PatientCase,
    ClinicalReport,
    build_clinical_graph,
)
from src.clinical_agent.evaluation import BENCHMARK_CASES


def mock_llm_responses(messages, **kwargs):
    content = messages[0]["content"] if messages else ""
    if "initial open differential diagnosis" in content.lower():
        return json.dumps({
            "differential": [
                {"condition": "Community-Acquired Pneumonia", "icd10_code": "J18.9", "tier": "HIGH_SUSPICION", "clinical_rationale": "Fever and cough."},
                {"condition": "Acute Bronchitis", "icd10_code": "J20.9", "tier": "MODERATE_SUSPICION", "clinical_rationale": "Cough illness."}
            ]
        })
    elif "available specialist tools to order:" in content.lower():
        import re
        match = re.search(r"Available Specialist Tools to Order:\s*\n?([^\n]+)", content)
        if match:
            avail_str = match.group(1).strip()
            avail_tools = [t.strip() for t in avail_str.split(",") if t.strip()]
            if "chest_xray_classifier" in avail_tools:
                chosen = "chest_xray_classifier"
            elif "laboratory_tests" in avail_tools:
                chosen = "laboratory_tests"
            elif avail_tools:
                chosen = "STOP"
            else:
                chosen = "STOP"
            return json.dumps({
                "selected_action": chosen,
                "clinical_indication": "Diagnostic evaluation.",
                "search_query": ""
            })
        return json.dumps({"selected_action": "STOP", "clinical_indication": "Workup complete.", "search_query": ""})
    elif "finalizing the definitive diagnostic consultation report" in content.lower():
        return json.dumps({
            "primary_diagnosis": "Community-Acquired Pneumonia",
            "disposition": "ADMIT_INPATIENT",
            "differential_diagnosis": [
                {"condition": "Community-Acquired Pneumonia", "icd10_code": "J18.9", "tier": "HIGH_SUSPICION", "clinical_rationale": "Dense consolidation confirmed on radiograph."}
            ],
            "clinical_impression": "Patient presents with acute symptoms and radiographic consolidation confirming bacterial pneumonia.",
            "recommendations": ["Initiate IV Ceftriaxone", "Monitor oxygenation"]
        })
    return "{}"



def test_agent_active_inquiry_flow():
    agent = AttendingClinicalAgent()
    case = BENCHMARK_CASES[0]
    
    with patch.object(agent.llm_client, "_call_completion", side_effect=mock_llm_responses):
        report = agent.diagnose(case=case, max_turns=3)
        
        assert isinstance(report, ClinicalReport)
        assert report.case_id == case.patient_id
        assert len(report.evidence_trail) >= 2
        assert len(report.differential_diagnosis) > 0
        assert report.primary_diagnosis == "Community-Acquired Pneumonia"
        assert report.clinical_impression != ""


def test_graph_state_machine_execution():
    agent = AttendingClinicalAgent()
    app = build_clinical_graph(agent)
    case = BENCHMARK_CASES[1]  # Heart failure case
    
    initial_state = {
        "case": case,
        "active_ddx": [],
        "evidence_history": [],
        "completed_tools": [],
        "selected_action": "",
        "clinical_indication": "",
        "search_query": None,
        "turn_count": 0,
        "max_turns": 3,
        "final_report": None,
    }
    
    with patch.object(agent.llm_client, "_call_completion", side_effect=mock_llm_responses):
        final_state = app.invoke(initial_state)
        report = final_state.get("final_report")
        
        assert report is not None
        assert isinstance(report, ClinicalReport)
        assert report.patient_id == case.patient_id
        assert len(report.evidence_trail) >= 2

