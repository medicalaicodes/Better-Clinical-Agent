"""
Clinical AI Agent: Simulated EHR Active Inquiry Benchmark Environment
Hides multi-modal patient data behind a blind interactive EHR interface.
Evaluates agent active diagnostic test selection, cost efficiency, and steps to convergence.

Academic Citations & Foundations:
- SDBench: "A Benchmark for Sequential Diagnosis with Language Models" (Microsoft Research, 2025; arXiv:2506.22405).
- MedAgent-Pro: "Towards Evidence-Based Multi-Modal Medical Diagnosis via Reasoning Agentic Workflow" (Wang et al., ICLR 2026; arXiv:2503.18968).
"""

from typing import Dict, List, Any, Optional, Tuple
from ..schemas import PatientCase, DiagnosticOrder, ToolResult, ClinicalReport
from ..agent import AttendingClinicalAgent


class SimulatedEHRCase:
    """Simulates an interactive electronic health record (EHR) with withheld diagnostic data."""

    def __init__(self, raw_case: Dict[str, Any]):
        self.case_id = raw_case.get("case_id", raw_case.get("patient_id", "PT_UNKNOWN"))
        self.ground_truth = raw_case.get("ground_truth_diagnosis", "")
        self.chief_complaint = raw_case.get("chief_complaint", "")
        self.vitals = raw_case.get("vitals", raw_case.get("initial_vitals", {}))
        self.age = raw_case.get("age", 50)
        self.gender = raw_case.get("gender", "Unknown")
        self._notes = raw_case.get("clinical_notes", "")
        self._image_path = raw_case.get("image_path")
        self._labs = raw_case.get("available_labs", {})

        self.ordered_tests: List[DiagnosticOrder] = []
        self.accumulated_cost_usd: float = 0.0

    def get_intake_view(self) -> PatientCase:
        """Returns baseline patient presentation without tests revealed."""
        return PatientCase(
            patient_id=self.case_id,
            age=self.age,
            gender=self.gender,
            chief_complaint=self.chief_complaint,
            vitals=self.vitals,
            clinical_notes=self._notes,
            image_path=None,
            available_labs={},
        )

    def get_full_case(self) -> PatientCase:
        """Returns complete patient case with all multimodal data."""
        return PatientCase(
            patient_id=self.case_id,
            age=self.age,
            gender=self.gender,
            chief_complaint=self.chief_complaint,
            vitals=self.vitals,
            clinical_notes=self._notes,
            image_path=self._image_path,
            available_labs=self._labs,
        )


class ActiveInquiryBenchmarkRunner:
    """Evaluates multi-turn active test ordering efficiency and diagnostic accuracy."""

    def __init__(self, agent: Optional[AttendingClinicalAgent] = None):
        self.agent = agent or AttendingClinicalAgent()

    def run_case_active(self, case: PatientCase, max_turns: int = 4) -> Dict[str, Any]:
        """Runs sequential active inquiry on a patient case and assesses accuracy."""
        report: ClinicalReport = self.agent.diagnose(case=case, max_turns=max_turns)

        top_pred = report.primary_diagnosis
        gt_raw = case.patient_id.lower()

        return {
            "case_id": case.patient_id,
            "top_1_prediction": top_pred,
            "disposition": report.disposition.value,
            "evidence_count": len(report.evidence_trail),
            "ordered_tools": [e.tool_name for e in report.evidence_trail],
            "final_report": report,
        }

