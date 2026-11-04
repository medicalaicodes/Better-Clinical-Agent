"""
Clinical AI Agent: Unit Tests for LangGraph State Machine Workflow
"""

import json
import pytest
from unittest.mock import patch
from src.clinical_agent.agent import AttendingClinicalAgent
from src.clinical_agent.schemas import PatientCase
from src.clinical_agent.schemas import ClinicalReport
from src.clinical_agent.graph import build_clinical_graph
from src.clinical_agent.evaluation import BENCHMARK_CASES


def mock_graph_llm(messages, **kwargs):
    content = messages[0]["content"] if messages else ""
    if "intake evaluation" in content.lower() or "initial open differential diagnosis" in content.lower():
        return json.dumps({
            "pertinent_positives": ["Fever", "Productive cough"],
            "pertinent_negatives": ["No orthopnea"],
            "differential": [
                {"condition": "Bacterial Pneumonia", "icd10_code": "J18.9", "tier": "HIGH_SUSPICION", "clinical_rationale": "High fever and productive cough."}
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
            else:
                chosen = "STOP"
            return json.dumps({
                "selected_action": chosen,
                "clinical_indication": "Diagnostic evaluation.",
                "search_query": ""
            })
        return json.dumps({"selected_action": "STOP", "clinical_indication": "Complete.", "search_query": ""})
    elif "finalizing the definitive diagnostic consultation report" in content.lower():
        return json.dumps({
            "primary_diagnosis": "Community-Acquired Pneumonia",
            "disposition": "ADMIT_INPATIENT",
            "differential_diagnosis": [
                {"condition": "Community-Acquired Pneumonia", "icd10_code": "J18.9", "tier": "HIGH_SUSPICION", "clinical_rationale": "Consolidation confirmed on chest radiography."}
            ],
            "clinical_impression": "Patient presents with acute symptoms consistent with pneumonia.",
            "recommendations": ["Initiate empiric antibiotics", "Monitor vital signs"]
        })
    return "{}"



def test_graph_state_machine_pneumonia_case():
    agent = AttendingClinicalAgent()
    app = build_clinical_graph(agent)
    case = BENCHMARK_CASES[0]  # Pneumonia case
    
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
    
    with patch.object(agent.llm_client, "_call_completion", side_effect=mock_graph_llm):
        final_state = app.invoke(initial_state)
        report = final_state.get("final_report")
        
        assert report is not None
        assert isinstance(report, ClinicalReport)
        assert report.patient_id == case.patient_id
        assert len(report.evidence_trail) >= 2
        assert report.primary_diagnosis != ""


def test_graph_state_machine_heart_failure_case():
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
    
    with patch.object(agent.llm_client, "_call_completion", side_effect=mock_graph_llm):
        final_state = app.invoke(initial_state)
        report = final_state.get("final_report")
        
        assert report is not None
        assert isinstance(report, ClinicalReport)
        assert report.patient_id == case.patient_id
        assert len(report.evidence_trail) >= 2

