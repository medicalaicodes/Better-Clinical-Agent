"""
Laboratory tool: compares each lab value with its normal range and names what an abnormal value
usually means. Ranges are adult reference ranges from the Tietz Textbook of Clinical Chemistry.
"""

from ..schemas import FindingItem, FindingStatus, ModalityType, ToolResult
from .base import BaseClinicalTool

# name -> (low, high, unit, meaning if high, meaning if low)
REFERENCE_RANGES = {
    "WBC": (4.5, 11.0, "x10^3/uL", "Leukocytosis (infection / inflammation / sepsis)", "Leukopenia"),
    "HEMOGLOBIN": (12.0, 17.5, "g/dL", "Polycythemia", "Anemia"),
    "PLATELETS": (150.0, 450.0, "x10^3/uL", "Thrombocytosis", "Thrombocytopenia"),
    "SODIUM": (135.0, 145.0, "mEq/L", "Hypernatremia", "Hyponatremia"),
    "POTASSIUM": (3.5, 5.0, "mEq/L", "Hyperkalemia", "Hypokalemia"),
    "BUN": (7.0, 20.0, "mg/dL", "Elevated BUN (renal impairment / dehydration)", "Low BUN"),
    "CREATININE": (0.6, 1.2, "mg/dL", "Acute kidney injury / elevated creatinine", "Low creatinine"),
    "GLUCOSE": (70.0, 99.0, "mg/dL", "Hyperglycemia", "Hypoglycemia"),
    "TROPONIN": (0.0, 0.04, "ng/mL", "Myocardial injury / elevated troponin", "Normal troponin"),
    "BNP": (0.0, 100.0, "pg/mL", "Marked elevation (acute decompensated heart failure)", "Normal BNP"),
    "NT_PROBNP": (0.0, 300.0, "pg/mL", "Marked elevation (ventricular wall stress / heart failure)", "Normal NT-proBNP"),
    "D_DIMER": (0.0, 0.50, "ug/mL FEU", "Elevated D-dimer (thromboembolism / PE / DVT)", "Normal D-dimer"),
    "PROCALCITONIN": (0.0, 0.25, "ug/L", "Bacterial infection / sepsis marker", "Low procalcitonin"),
    "CRP": (0.0, 10.0, "mg/L", "Acute inflammation", "Normal CRP"),
    "LACTATE": (0.5, 2.0, "mmol/L", "Lactic acidosis / poor tissue perfusion", "Normal lactate"),
}


class LaboratoryTestsTool(BaseClinicalTool):
    name = "laboratory_tests"
    description = ("Blood test analyzer: compares blood count, kidney, cardiac (troponin, BNP), "
                   "inflammatory (CRP, procalcitonin), D-dimer and lactate values with normal ranges.")
    modality = ModalityType.LABORATORY

    def run(self, case, **options):
        labs = options.get("labs") or case.available_labs
        if not labs:
            return ToolResult(tool_name=self.name, modality=self.modality, findings=[],
                              summary="No laboratory values available for this patient.",
                              raw_output={"status": "no_labs"})

        findings = []
        abnormal = []
        for lab_name, value in labs.items():
            key = lab_name.upper().replace("-", "_").replace(" ", "_")

            if key not in REFERENCE_RANGES:
                # A lab we have no range for: report the value without judging it.
                findings.append(FindingItem(name=f"Lab: {lab_name}", quantitative_value=float(value), unit=""))
                continue

            low, high, unit, meaning_if_high, meaning_if_low = REFERENCE_RANGES[key]

            if value > high:
                direction = "HIGH"
                meaning = meaning_if_high
                name = f"Elevated {key} ({meaning}): {value} {unit}"
            elif value < low:
                direction = "LOW"
                meaning = meaning_if_low
                name = f"Low {key} ({meaning}): {value} {unit}"
            else:
                direction = "NORMAL"
                meaning = "Within normal limits"
                name = f"Normal {key}: {value} {unit}"

            if direction == "NORMAL":
                status = FindingStatus.ABSENT        # a normal value means the abnormality is absent
            else:
                status = FindingStatus.PRESENT
                abnormal.append(f"{key} {value} {unit} [{direction}]")

            findings.append(FindingItem(name=name, status=status, quantitative_value=float(value), unit=unit,
                                        details={"reference_range": f"{low}-{high} {unit}",
                                                 "interpretation": meaning}))

        if abnormal:
            summary = f"Laboratory evaluation found {len(abnormal)} abnormal value(s): {'; '.join(abnormal)}."
        else:
            summary = "All laboratory values are within their normal reference ranges."

        return ToolResult(tool_name=self.name, modality=self.modality, findings=findings, summary=summary,
                          raw_output={"labs": labs, "abnormal_count": len(abnormal)})
