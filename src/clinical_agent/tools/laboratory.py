"""
Clinical AI Agent: Emergency Laboratory Specialist Tool

Academic Citations & Foundations:
- Clinical Reference Ranges and Diagnostic Biomarker Interpretation:
  * Rifai et al., "Tietz Textbook of Clinical Chemistry and Molecular Diagnostics" (6th Edition, Elsevier, 2018).
  * Maisel et al., "Rapid measurement of B-type natriuretic peptide in the emergency diagnosis of heart failure" (NEJM 2002; DOI:10.1056/NEJMoa020233).
  * Schuetz et al., "Procalcitonin to initiate or discontinue antibiotics in acute respiratory tract infections" (Cochrane Database of Systematic Reviews, 2017).
"""

from typing import Dict, List, Optional, Any
from .base import BaseClinicalTool
from ..schemas import PatientCase, ToolResult, ModalityType, FindingItem, FindingStatus


# Clinical Reference Ranges based on Tietz Textbook of Clinical Chemistry
LAB_REFERENCE_RANGES: Dict[str, Dict[str, Any]] = {
    "WBC": {"low": 4.5, "high": 11.0, "unit": "x10^3/uL", "clinical_name": "White Blood Cell Count", "high_label": "Leukocytosis (Infection / Inflammation / Sepsis)", "low_label": "Leukopenia"},
    "HEMOGLOBIN": {"low": 12.0, "high": 17.5, "unit": "g/dL", "clinical_name": "Hemoglobin", "high_label": "Polycythemia", "low_label": "Anemia"},
    "PLATELETS": {"low": 150.0, "high": 450.0, "unit": "x10^3/uL", "clinical_name": "Platelets", "high_label": "Thrombocytosis", "low_label": "Thrombocytopenia"},
    "SODIUM": {"low": 135.0, "high": 145.0, "unit": "mEq/L", "clinical_name": "Serum Sodium", "high_label": "Hypernatremia", "low_label": "Hyponatremia"},
    "POTASSIUM": {"low": 3.5, "high": 5.0, "unit": "mEq/L", "clinical_name": "Serum Potassium", "high_label": "Hyperkalemia", "low_label": "Hypokalemia"},
    "BUN": {"low": 7.0, "high": 20.0, "unit": "mg/dL", "clinical_name": "Blood Urea Nitrogen", "high_label": "Elevated BUN (Renal Impairment / Dehydration)", "low_label": "Low BUN"},
    "CREATININE": {"low": 0.6, "high": 1.2, "unit": "mg/dL", "clinical_name": "Serum Creatinine", "high_label": "Acute Kidney Injury / Elevated Creatinine", "low_label": "Low Creatinine"},
    "GLUCOSE": {"low": 70.0, "high": 99.0, "unit": "mg/dL", "clinical_name": "Fasting Glucose", "high_label": "Hyperglycemia", "low_label": "Hypoglycemia"},
    "TROPONIN": {"low": 0.0, "high": 0.04, "unit": "ng/mL", "clinical_name": "Cardiac Troponin I", "high_label": "Myocardial Injury / Elevated Troponin", "low_label": "Normal Troponin"},
    "BNP": {"low": 0.0, "high": 100.0, "unit": "pg/mL", "clinical_name": "B-type Natriuretic Peptide", "high_label": "Marked Elevation (Acute Decompensated Heart Failure)", "low_label": "Normal BNP"},
    "NT_PROBNP": {"low": 0.0, "high": 300.0, "unit": "pg/mL", "clinical_name": "N-Terminal pro-BNP", "high_label": "Marked Elevation (Ventricular Wall Stress / Heart Failure)", "low_label": "Normal NT-proBNP"},
    "D_DIMER": {"low": 0.0, "high": 0.50, "unit": "ug/mL FEU", "clinical_name": "D-Dimer", "high_label": "Elevated D-Dimer (Thromboembolism / PE / DVT)", "low_label": "Normal D-Dimer (Excludes PE)"},
    "PROCALCITONIN": {"low": 0.0, "high": 0.25, "unit": "ug/L", "clinical_name": "Procalcitonin", "high_label": "Bacterial Infection / Sepsis Marker", "low_label": "Low Procalcitonin"},
    "CRP": {"low": 0.0, "high": 10.0, "unit": "mg/L", "clinical_name": "C-Reactive Protein", "high_label": "Acute Phase Reactant / Inflammatory Marker", "low_label": "Normal CRP"},
    "LACTATE": {"low": 0.5, "high": 2.0, "unit": "mmol/L", "clinical_name": "Serum Lactate", "high_label": "Lactic Acidosis / Tissue Hypoperfusion", "low_label": "Normal Lactate"},
}



class LaboratoryTestsTool(BaseClinicalTool):
    name: str = "laboratory_tests"
    description: str = (
        "Evaluates quantitative emergency laboratory biomarker panels including "
        "Complete Blood Count (WBC, Hb, Plt), Comprehensive Metabolic Panel (BMP, Creatinine, BUN), "
        "Cardiac Biomarkers (Troponin, BNP, NT-proBNP), Inflammatory Markers (Procalcitonin, CRP), and D-Dimer."
    )
    modality: ModalityType = ModalityType.LABORATORY

    def run(self, case: PatientCase, **kwargs) -> ToolResult:
        labs = kwargs.get("labs") or case.available_labs
        if not labs:
            return ToolResult(
                tool_name=self.name,
                modality=self.modality,
                findings=[],
                summary="No laboratory biomarker values available for this patient.",
                raw_output={"status": "no_labs_ordered"},
            )

        findings: List[FindingItem] = []
        abnormal_summaries: List[str] = []

        for lab_key, value in labs.items():
            norm_key = lab_key.upper().replace("-", "_").replace(" ", "_")
            ref_info = LAB_REFERENCE_RANGES.get(norm_key)

            if not ref_info:
                findings.append(
                    FindingItem(
                        name=f"Lab: {lab_key}",
                        status=FindingStatus.PRESENT,
                        confidence=1.0,
                        quantitative_value=float(value),
                        unit="",
                    )
                )
                continue

            low_val = ref_info["low"]
            high_val = ref_info["high"]
            unit_str = ref_info["unit"]
            clin_name = ref_info["clinical_name"]

            if value > high_val:
                label = ref_info["high_label"]
                status = FindingStatus.PRESENT
                dev_sev = round(abs(value - high_val) / high_val, 3) if high_val > 0 else 1.0
                abnormal_summaries.append(f"{norm_key} {value} {unit_str} [HIGH, dev: +{dev_sev*100:.0f}%]")
                findings.append(
                    FindingItem(
                        name=f"Elevated {norm_key} ({label}): {value} {unit_str}",
                        status=status,
                        confidence=1.0,
                        quantitative_value=float(value),
                        unit=unit_str,
                        details={
                            "reference_range": f"{low_val}-{high_val} {unit_str}",
                            "interpretation": label,
                            "deviation_severity": dev_sev,
                            "is_abnormal": True,
                        },
                    )
                )
            elif value < low_val:
                label = ref_info["low_label"]
                status = FindingStatus.PRESENT
                dev_sev = round(abs(low_val - value) / low_val, 3) if low_val > 0 else 1.0
                abnormal_summaries.append(f"{norm_key} {value} {unit_str} [LOW, dev: -{dev_sev*100:.0f}%]")
                findings.append(
                    FindingItem(
                        name=f"Low {norm_key} ({label}): {value} {unit_str}",
                        status=status,
                        confidence=1.0,
                        quantitative_value=float(value),
                        unit=unit_str,
                        details={
                            "reference_range": f"{low_val}-{high_val} {unit_str}",
                            "interpretation": label,
                            "deviation_severity": dev_sev,
                            "is_abnormal": True,
                        },
                    )
                )
            else:
                findings.append(
                    FindingItem(
                        name=f"Normal {norm_key}: {value} {unit_str}",
                        status=FindingStatus.ABSENT,
                        confidence=1.0,
                        quantitative_value=float(value),
                        unit=unit_str,
                        details={
                            "reference_range": f"{low_val}-{high_val} {unit_str}",
                            "interpretation": "Within physiological limits",
                            "is_abnormal": False,
                        },
                    )
                )

        if abnormal_summaries:
            summary = f"Laboratory evaluation revealed {len(abnormal_summaries)} abnormal biomarker(s): {'; '.join(abnormal_summaries)}."
        else:
            summary = "All evaluated laboratory biomarker parameters are within normal physiological reference ranges."

        return ToolResult(
            tool_name=self.name,
            modality=self.modality,
            findings=findings,
            summary=summary,
            raw_output={"tested_biomarkers": labs, "abnormal_count": len(abnormal_summaries)},
        )
