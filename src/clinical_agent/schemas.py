"""
The data the clinical agent passes around. They are pydantic models: like plain classes, but
pydantic checks the types when an object is created (an age of "abc" is refused straight away).
"""

from enum import Enum

from pydantic import BaseModel


class ModalityType(str, Enum):
    IMAGING = "imaging"
    SEGMENTATION = "segmentation"
    LABORATORY = "laboratory"
    KNOWLEDGE = "knowledge"
    CLINICAL_NOTES = "clinical_notes"


class FindingStatus(str, Enum):
    PRESENT = "present"
    ABSENT = "absent"


class SuspicionTier(str, Enum):
    HIGH_SUSPICION = "HIGH_SUSPICION"
    MODERATE_SUSPICION = "MODERATE_SUSPICION"
    RULE_OUT = "RULE_OUT"


class DispositionType(str, Enum):
    ADMIT_INPATIENT = "ADMIT_INPATIENT"
    DISCHARGE_WITH_FOLLOWUP = "DISCHARGE_WITH_FOLLOWUP"
    ESCALATE_TO_PHYSICIAN = "ESCALATE_TO_PHYSICIAN"


class PatientCase(BaseModel):
    """Everything we know about the patient at the start."""
    patient_id: str
    age: int
    gender: str
    chief_complaint: str
    vitals: dict = {}               # e.g. {"HR": 112, "BP": "124/78", "SpO2": 90}
    clinical_notes: str = ""
    image_path: str | None = None   # chest X-ray, if there is one
    available_labs: dict = {}       # e.g. {"WBC": 17.8, "CRP": 64.0}


class FindingItem(BaseModel):
    """One finding a tool reported, e.g. 'Elevated WBC' or 'CXR: Consolidation'."""
    name: str
    status: FindingStatus = FindingStatus.PRESENT
    confidence: float = 1.0
    quantitative_value: float | None = None
    unit: str | None = None
    details: dict = {}


class ToolResult(BaseModel):
    """What one tool returns: its findings, a one-paragraph summary, and its raw output."""
    tool_name: str
    modality: ModalityType
    findings: list[FindingItem] = []
    summary: str
    raw_output: dict = {}


class DifferentialHypothesis(BaseModel):
    """One candidate diagnosis on the agent's list."""
    condition: str
    tier: SuspicionTier = SuspicionTier.MODERATE_SUSPICION
    clinical_rationale: str


class ClinicalReport(BaseModel):
    """The final report the agent writes."""
    case_id: str
    patient_id: str
    primary_diagnosis: str
    disposition: DispositionType
    differential_diagnosis: list[DifferentialHypothesis] = []
    key_findings: list[FindingItem] = []
    clinical_impression: str
    recommendations: list[str] = []
    evidence_trail: list[ToolResult] = []
