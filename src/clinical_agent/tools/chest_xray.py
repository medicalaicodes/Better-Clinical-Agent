"""
Clinical AI Agent: Chest X-Ray Classifier Specialist Tool

Academic Citations & Foundations:
- Deep Learning for Chest Radiograph Pathology Classification:
  * Cohen et al., "TorchXRayVision: A library of chest X-ray datasets and models" (MIDL 2022; arXiv:2111.00595).
  * Rajpurkar et al., "CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning" (Stanford University, 2017; arXiv:1711.05225).
  * MedRAX: "A Medical Reasoning Agent for Chest X-ray" (Bo Wang Lab, 2025; arXiv:2502.02618).
"""

import os
from typing import Dict, List, Optional, Any

from .base import BaseClinicalTool
from ..schemas import PatientCase, ToolResult, ModalityType, FindingItem, FindingStatus
from ..models import DenseNetCXRModel


class ChestXRayClassifierTool(BaseClinicalTool):
    name: str = "chest_xray_classifier"
    description: str = (
        "Analyzes chest radiographs (CXR) using a deep DenseNet-121 model. "
        "Outputs predicted probabilities across 18 thoracic pathologies including "
        "Consolidation, Lung Opacity, Effusion, Cardiomegaly, Pneumothorax, Edema, and Atelectasis."
    )
    modality: ModalityType = ModalityType.IMAGING

    def __init__(self, device: Optional[str] = None):
        self.device = device
        self._model_wrapper: Optional[DenseNetCXRModel] = None

    def _get_model(self) -> DenseNetCXRModel:
        if self._model_wrapper is None:
            self._model_wrapper = DenseNetCXRModel(device=self.device)
        return self._model_wrapper

    def run(self, case: PatientCase, **kwargs) -> ToolResult:
        image_path = kwargs.get("image_path") or case.image_path
        if not image_path or not os.path.exists(image_path):
            return ToolResult(
                tool_name=self.name,
                modality=self.modality,
                findings=[],
                summary="No chest radiograph image available for this patient.",
                raw_output={"status": "image_not_found"},
            )

        # Execute TorchXRayVision DenseNet inference directly
        model = self._get_model()
        probabilities = model.predict(image_path)
        thresholds = model.op_threshs

        # Zero-centered clinical certainty scaling: 0.50 = 0% baseline (unsure), 0.60 = +20% present, 0.40 = 20% absent
        findings: List[FindingItem] = []
        positive_pathologies = []

        for path_name, prob in sorted(probabilities.items(), key=lambda x: x[1], reverse=True):
            thresh = thresholds.get(path_name, 0.50)
            # Scaled certainty: (prob - 0.50) * 2
            certainty_score = round(float((prob - 0.50) * 2.0), 3)
            certainty_pct = round(certainty_score * 100.0, 1)

            # Clinically meaningful positive signal requires certainty >= +30% (prob >= 0.65)
            is_present = certainty_score >= 0.30

            if is_present:
                positive_pathologies.append((path_name, prob, certainty_pct))
                findings.append(
                    FindingItem(
                        name=f"CXR: {path_name}",
                        status=FindingStatus.PRESENT,
                        confidence=prob,
                        quantitative_value=prob,
                        unit="probability",
                        details={
                            "operating_threshold": thresh,
                            "certainty_pct": f"{certainty_pct:+.1f}%",
                            "is_positive": True,
                        },
                    )
                )

        if positive_pathologies:
            lead_name, lead_prob, lead_cert = positive_pathologies[0]
            summary_items = [f"{p} (Prob: {pr:.3f}, Certainty: {cert:+.1f}%)" for p, pr, cert in positive_pathologies[:3]]
            summary = f"Chest radiograph demonstrates positive radiographic signs: {', '.join(summary_items)}."
        else:
            summary = "Chest radiograph demonstrates clear lung fields bilaterally without evidence of focal consolidation, pneumothorax, pulmonary edema, or pleural effusion."
            findings.append(
                FindingItem(
                    name="CXR: Clear Lung Fields / No Focal Acute Pathology",
                    status=FindingStatus.PRESENT,
                    confidence=0.95,
                    details={"certainty": "Clear bilaterally"},
                )
            )


        return ToolResult(
            tool_name=self.name,
            modality=self.modality,
            findings=findings,
            summary=summary,
            raw_output={"probabilities": probabilities, "image_path": image_path},
        )
