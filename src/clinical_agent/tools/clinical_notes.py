"""
Clinical AI Agent: Clinical Notes & History Extractor Tool

Academic Citations & Foundations:
- Clinical NLP & Electronic Health Record Free-Text Processing:
  * Wang et al., "MedAgent-Pro: Towards Evidence-Based Multi-Modal Medical Diagnosis via Reasoning Agentic Workflow" (ICLR 2026; arXiv:2503.18968) - Qualitative symptom extraction.
  * SDBench: "A Benchmark for Sequential Diagnosis with Language Models" (Microsoft Research, 2025; arXiv:2506.22405) - Patient intake parsing.
"""

import re
from typing import Dict, List, Optional, Any
from .base import BaseClinicalTool
from ..schemas import PatientCase, ToolResult, ModalityType, FindingItem, FindingStatus


class ClinicalNotesExtractorTool(BaseClinicalTool):
    name: str = "clinical_notes_extractor"
    description: str = (
        "Parses unstructured triage notes, emergency department physician documentation, "
        "and physical examination records to extract structured symptom profiles, exam signs, and risk factors."
    )
    modality: ModalityType = ModalityType.CLINICAL_NOTES

    def run(self, case: PatientCase, **kwargs) -> ToolResult:
        notes = kwargs.get("notes") or case.clinical_notes
        if not notes:
            return ToolResult(
                tool_name=self.name,
                modality=self.modality,
                findings=[],
                summary="No clinical notes or triage documentation available.",
                raw_output={"status": "no_notes"},
            )

        findings: List[FindingItem] = []
        notes_lower = notes.lower()

        # Extract documented symptoms & exam signs
        symptom_patterns = {
            "Fever / Pyrexia": r"(fever|pyrexia|febrile|rigors|chills)",
            "Crackles / Rales": r"(crackles|rales|crepitations)",
            "Bronchial Breath Sounds": r"(bronchial breath sounds|consolidation sounds)",
            "Wheezing": r"(wheeze|wheezing|rhonchi)",
            "Jugular Venous Distension (JVD)": r"(jvd|jugular venous|neck vein distension)",
            "Orthopnea": r"(orthopnea|pillows to sleep|pnd|paroxysmal nocturnal)",
            "Pedal / Peripheral Edema": r"(pedal edema|pitting edema|leg swelling|ankle edema)",
            "Chest Wall Point Tenderness": r"(point tenderness|palpation tenderness|costochondral)",
            "Pleuritic Chest Pain": r"(pleuritic|worse on inspiration|sharp with breathing)",
            "Recent Immobilization / Travel": r"(flight|travel|immobilization|bedrest|cast)",
            "Purulent Sputum": r"(purulent|green sputum|yellow sputum|productive cough)",
        }

        extracted_signs = []
        for sign_name, pattern in symptom_patterns.items():
            if re.search(pattern, notes_lower):
                extracted_signs.append(sign_name)
                findings.append(
                    FindingItem(
                        name=f"Exam/History Sign: {sign_name}",
                        status=FindingStatus.PRESENT,
                        confidence=0.95,
                        details={"documentation": notes},
                    )
                )

        if extracted_signs:
            summary = f"Extracted {len(extracted_signs)} clinical exam/history features: {', '.join(extracted_signs)}."
        else:
            summary = f"Clinical history reviewed: {notes[:120]}..."

        return ToolResult(
            tool_name=self.name,
            modality=self.modality,
            findings=findings,
            summary=summary,
            raw_output={"extracted_signs": extracted_signs, "raw_notes": notes},
        )
