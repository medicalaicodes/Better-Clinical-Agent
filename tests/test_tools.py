"""Tests for the clinical agent's tools. The image tests are skipped if their libraries are missing."""

import pytest
from src.clinical_agent.schemas import ModalityType, FindingStatus
from src.clinical_agent.tools import (
    ChestXRayClassifierTool,
    MedSAMSegmenterTool,
    LaboratoryTestsTool,
    MedicalKnowledgeRAGTool,
)
from src.clinical_agent.evaluation.sample_cases import BENCHMARK_CASES


def test_cxr_classifier_tool():
    pytest.importorskip("torchxrayvision")
    tool = ChestXRayClassifierTool()
    case = BENCHMARK_CASES[0]
    result = tool.run(case)
    
    assert result.tool_name == "chest_xray_classifier"
    assert result.modality == ModalityType.IMAGING
    assert len(result.findings) > 0
    assert result.summary != ""
    # outputs are calibrated scores in [0, 1], not double-sigmoided values stuck in [0.5, 0.73]
    probabilities = result.raw_output["probabilities"]
    assert min(probabilities.values()) < 0.5


def test_medsam_segmenter_tool():
    pytest.importorskip("transformers")
    tool = MedSAMSegmenterTool()
    case = BENCHMARK_CASES[0]
    result = tool.run(case)
    
    assert result.tool_name == "medsam_segmenter"
    assert result.modality == ModalityType.SEGMENTATION
    assert len(result.findings) >= 2
    names = [finding.name for finding in result.findings]
    assert "MedSAM: Left Lung Field Segmentation Area" in names


def test_laboratory_tests_tool():
    tool = LaboratoryTestsTool()
    case = BENCHMARK_CASES[0]
    result = tool.run(case)
    
    assert result.tool_name == "laboratory_tests"
    assert result.modality == ModalityType.LABORATORY
    abnormal = []
    for finding in result.findings:
        if finding.status == FindingStatus.PRESENT and (finding.name.startswith("Elevated") or finding.name.startswith("Low")):
            abnormal.append(finding)
    assert len(abnormal) >= 3


def test_medical_knowledge_tool_uses_the_retriever(toy_retriever):
    tool = MedicalKnowledgeRAGTool(retriever=toy_retriever, k=2)
    result = tool.run(BENCHMARK_CASES[2], query="pulmonary embolism D-dimer pretest probability")

    assert result.tool_name == "medical_knowledge_rag"
    assert result.modality == ModalityType.KNOWLEDGE
    assert len(result.findings) == 2
    assert result.raw_output["snippet_ids"][0].startswith("PE_")
