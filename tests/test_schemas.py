"""
Clinical AI Agent: Data Schema Unit Tests
"""

import pytest
from src.clinical_agent.schemas import (
    ModalityType,
    FindingStatus,
    FindingItem,
    ToolResult,
    PatientCase,
    ClinicalReport,
    DispositionType,
)


def test_finding_item_instantiation():
    finding = FindingItem(
        name="Lung Opacity",
        status=FindingStatus.PRESENT,
        confidence=0.88,
        quantitative_value=0.88,
        unit="probability",
    )
    assert finding.name == "Lung Opacity"
    assert finding.status == FindingStatus.PRESENT
    assert finding.confidence == 0.88


def test_patient_case_validation():
    case = PatientCase(
        patient_id="PT_TEST_01",
        age=55,
        gender="Male",
        chief_complaint="Chest pain and dyspnea",
        vitals={"HR": 95, "BP": "130/85"},
    )
    assert case.patient_id == "PT_TEST_01"
    assert case.age == 55
    assert case.gender == "Male"


def test_tool_result_creation():
    result = ToolResult(
        tool_name="test_tool",
        modality=ModalityType.LABORATORY,
        findings=[
            FindingItem(name="Leukocytosis", status=FindingStatus.PRESENT, confidence=0.99)
        ],
        summary="High WBC count observed.",
    )
    assert result.tool_name == "test_tool"
    assert len(result.findings) == 1


def test_clinical_report_structure():
    report = ClinicalReport(
        case_id="PT_001",
        patient_id="PT_001",
        primary_diagnosis="Community-Acquired Pneumonia",
        disposition=DispositionType.ADMIT_INPATIENT,
        clinical_impression="Patient has typical bacterial lobar consolidation.",
        recommendations=["Start IV Ceftriaxone", "Monitor oxygenation"],
    )
    assert report.primary_diagnosis == "Community-Acquired Pneumonia"
    assert report.disposition == DispositionType.ADMIT_INPATIENT
    assert len(report.recommendations) == 2
