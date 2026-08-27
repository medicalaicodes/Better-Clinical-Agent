"""
Clinical AI Agent: Specialist Tools Test Suite
"""

import pytest
from src.clinical_agent.schemas import ModalityType, FindingStatus
from src.clinical_agent.tools import (
    ChestXRayClassifierTool,
    MedSAMSegmenterTool,
    LaboratoryTestsTool,
    MedicalKnowledgeRAGTool,
    ClinicalNotesExtractorTool,
)
from src.clinical_agent.evaluation.sample_cases import BENCHMARK_CASES


def test_cxr_classifier_tool():
    tool = ChestXRayClassifierTool()
    case = BENCHMARK_CASES[0]
    result = tool.run(case)
    
    assert result.tool_name == "chest_xray_classifier"
    assert result.modality == ModalityType.IMAGING
    assert len(result.findings) > 0
    assert result.summary != ""


def test_medsam_segmenter_tool():
    tool = MedSAMSegmenterTool()
    case = BENCHMARK_CASES[0]
    result = tool.run(case)
    
    assert result.tool_name == "medsam_segmenter"
    assert result.modality == ModalityType.SEGMENTATION
    assert len(result.findings) >= 2
    # Area and symmetry metrics should be present
    area_findings = [f for f in result.findings if "Area" in f.name]
    assert len(area_findings) >= 2


def test_laboratory_tests_tool():
    tool = LaboratoryTestsTool()
    case = BENCHMARK_CASES[0]
    result = tool.run(case)
    
    assert result.tool_name == "laboratory_tests"
    assert result.modality == ModalityType.LABORATORY
    abnormal_findings = [f for f in result.findings if f.status == FindingStatus.PRESENT and ("Elevated" in f.name or "Low" in f.name)]
    assert len(abnormal_findings) >= 3


def test_medical_knowledge_rag_tool():
    tool = MedicalKnowledgeRAGTool()
    case = BENCHMARK_CASES[0]
    result = tool.run(case, query="Community-Acquired Pneumonia")
    
    assert result.tool_name == "medical_knowledge_rag"
    assert result.modality == ModalityType.KNOWLEDGE
    assert len(result.findings) > 0
    assert "guideline" in result.summary.lower() or "clinical" in result.summary.lower()


def test_medical_knowledge_rag_rigorous_retrieval():
    """Rigorously tests semantic vector retrieval across diverse clinical presentations without data leakage."""
    tool = MedicalKnowledgeRAGTool()
    
    test_queries = [
        ("high fever, diarrhea, confusion, relative bradycardia, hyponatremia", "Legionnaires' Disease"),
        ("severe orthopnea, elevated jugular venous distension, cardiomegaly, bnp 1680", "Heart Failure"),
        ("sudden sharp pleuritic chest pain, calf swelling after international flight, d-dimer", "Pulmonary Embolism"),
        ("subacute dry nonproductive cough, severe hypoxemia in an immunocompromised HIV patient with low CD4", "Pneumocystis"),
        ("currant-jelly sputum, upper lobe cavitation and bulging fissure in diabetic patient", "Klebsiella"),
        ("sharp parasternal point tenderness reproducible on direct palpation", "Costochondritis"),
        ("fever, anosmia, loss of taste, bilateral ground glass opacities and lymphopenia", "COVID-19"),
    ]
    
    for query_str, expected_disease_substring in test_queries:
        matches = tool.semantic_search(query_str, top_k=1)
        assert len(matches) > 0
        top_doc, score = matches[0]
        assert score > 0.15, f"Low similarity score ({score}) for query: '{query_str}'"
        assert expected_disease_substring.lower() in top_doc["disease"].lower(), (
            f"Expected '{expected_disease_substring}' to match query '{query_str}', but got '{top_doc['disease']}' with score {score}"
        )


def test_clinical_notes_extractor_tool():
    tool = ClinicalNotesExtractorTool()
    case = BENCHMARK_CASES[0]
    result = tool.run(case)
    
    assert result.tool_name == "clinical_notes_extractor"
    assert result.modality == ModalityType.CLINICAL_NOTES
    assert len(result.findings) > 0
    finding_names = [f.name.lower() for f in result.findings]
    assert any("fever" in name or "pyrexia" in name for name in finding_names)
