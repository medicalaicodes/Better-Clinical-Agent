"""
Clinical AI Agent: Core Data Schemas

Academic Citations & Foundations:
- Diagnostic action, observation state, and multi-turn inquiry structures inspired by:
  * SDBench & MAI-DxO: "A Benchmark for Sequential Diagnosis with Language Models" (Microsoft Research, 2025; arXiv:2506.22405).
  * MedAgent-Pro: "Towards Evidence-Based Multi-Modal Medical Diagnosis via Reasoning Agentic Workflow" (Wang et al., ICLR 2026; arXiv:2503.18968).
  * MedRAX: "A Medical Reasoning Agent for Chest X-ray" (Bo Wang Lab, 2025; arXiv:2502.02618).
"""

from enum import Enum
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, ConfigDict


class ModalityType(str, Enum):
    IMAGING = "imaging"
    SEGMENTATION = "segmentation"
    LABORATORY = "laboratory"
    KNOWLEDGE = "knowledge"
    CLINICAL_NOTES = "clinical_notes"


class FindingStatus(str, Enum):
    PRESENT = "present"
    ABSENT = "absent"
    UNCERTAIN = "uncertain"


class SuspicionTier(str, Enum):
    HIGH_SUSPICION = "HIGH_SUSPICION"
    MODERATE_SUSPICION = "MODERATE_SUSPICION"
    RULE_OUT = "RULE_OUT"


class DispositionType(str, Enum):
    ADMIT_INPATIENT = "ADMIT_INPATIENT"
    DISCHARGE_WITH_FOLLOWUP = "DISCHARGE_WITH_FOLLOWUP"
    ESCALATE_TO_PHYSICIAN = "ESCALATE_TO_PHYSICIAN"


class FindingItem(BaseModel):
    """Atomic structured clinical finding extracted by a specialist tool."""
    model_config = ConfigDict(frozen=True)

    name: str = Field(..., description="Clinical finding name (e.g., Lung Opacity, Leukocytosis)")
    status: FindingStatus = Field(default=FindingStatus.PRESENT, description="Presence status")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Model confidence score")
    quantitative_value: Optional[float] = Field(default=None, description="Numerical metric if applicable")
    unit: Optional[str] = Field(default=None, description="Measurement unit (e.g., mg/L, pg/mL)")
    details: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata or context")


class ToolResult(BaseModel):
    """Standardized result packet returned by any specialist diagnostic tool."""
    tool_name: str = Field(..., description="Name of the specialist tool that produced this result")
    modality: ModalityType = Field(..., description="Clinical data modality")
    findings: List[FindingItem] = Field(default_factory=list, description="Extracted clinical findings")
    summary: str = Field(..., description="Natural language summary of the tool output")
    raw_output: Dict[str, Any] = Field(default_factory=dict, description="Raw model or tool outputs")


class DifferentialHypothesis(BaseModel):
    """A diagnostic candidate condition in the physician's differential."""
    condition: str = Field(..., description="Disease or pathology name (e.g., Community-Acquired Pneumonia)")
    icd10_code: Optional[str] = Field(default=None, description="Standard ICD-10 code if available")
    tier: SuspicionTier = Field(default=SuspicionTier.MODERATE_SUSPICION, description="Clinical suspicion tier")
    clinical_rationale: str = Field(..., description="Clinical reasoning explaining the hypothesis")


class DiagnosticOrder(BaseModel):
    """An action or test requested by the attending physician agent."""
    order_id: str = Field(..., description="Unique identifier for the diagnostic order")
    tool_name: str = Field(..., description="Target specialist tool name")
    display_name: str = Field(..., description="Human-readable test name (e.g., Chest X-Ray)")
    indication: str = Field(..., description="Medical reason for requesting this test")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Tool execution parameters")


class PatientCase(BaseModel):
    """Complete patient presentation with raw multimodal diagnostic data."""
    patient_id: str = Field(..., description="Unique patient identifier")
    age: int = Field(..., ge=0, le=120, description="Patient age in years")
    gender: str = Field(..., description="Patient biological sex")
    chief_complaint: str = Field(..., description="Primary presenting complaint")
    vitals: Dict[str, Any] = Field(default_factory=dict, description="Vital signs (HR, BP, Temp, SpO2, RR)")
    clinical_notes: str = Field(default="", description="Triage history and physical exam notes")
    image_path: Optional[str] = Field(default=None, description="Path to radiographic or photographic image")
    available_labs: Dict[str, float] = Field(default_factory=dict, description="Available laboratory biomarker values")


class ClinicalReport(BaseModel):
    """Comprehensive final consultation report synthesized by the attending agent."""
    case_id: str = Field(..., description="Case identifier")
    patient_id: str = Field(..., description="Patient identifier")
    primary_diagnosis: str = Field(..., description="Lead clinical diagnosis")
    disposition: DispositionType = Field(..., description="Recommended triage disposition")
    differential_diagnosis: List[DifferentialHypothesis] = Field(default_factory=list, description="Ranked differential")
    key_findings: List[FindingItem] = Field(default_factory=list, description="All gathered clinical findings")
    clinical_impression: str = Field(..., description="Synthesized diagnostic impression prose")
    recommendations: List[str] = Field(default_factory=list, description="Actionable clinical management steps")
    evidence_trail: List[ToolResult] = Field(default_factory=list, description="Chronological record of tool outputs")
