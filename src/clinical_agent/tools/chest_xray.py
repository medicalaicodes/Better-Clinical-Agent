"""
Clinical AI Agent: Chest X-Ray Classifier Specialist Tool

Academic Citations & Foundations:
- Deep Learning for Chest Radiograph Pathology Classification:
  * Cohen et al., "TorchXRayVision: A library of chest X-ray datasets and models" (MIDL 2022; arXiv:2111.00595).
  * Rajpurkar et al., "CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning" (Stanford University, 2017; arXiv:1711.05225).
  * MedRAX: "A Medical Reasoning Agent for Chest X-ray" (Bo Wang Lab, 2025; arXiv:2502.02618).
"""

import os
from typing import List, Optional

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

        # Outputs are operating-point calibrated: 0.5 is the threshold TorchXRayVision chose
        # per label on its validation data, so ">= 0.5" means "positive at that operating point".
        model = self._get_model()
        probabilities = model.predict(image_path)

        findings: List[FindingItem] = []
        positives = []
        for label, prob in sorted(probabilities.items(), key=lambda x: x[1], reverse=True):
            if prob < 0.5:
                continue
            positives.append((label, prob))
            findings.append(
                FindingItem(
                    name=f"CXR: {label}",
                    status=FindingStatus.PRESENT,
                    confidence=prob,
                    quantitative_value=prob,
                    unit="calibrated score (0.5 = operating point)",
                    details={"raw_operating_threshold": model.op_threshs.get(label)},
                )
            )

        if positives:
            items = [f"{label} ({prob:.2f})" for label, prob in positives[:4]]
            summary = (f"Chest radiograph classifier is positive (score >= 0.5 operating point) for: "
                       f"{', '.join(items)}.")
        else:
            top_label, top_prob = max(probabilities.items(), key=lambda x: x[1])
            summary = (f"Chest radiograph classifier is negative for all {len(probabilities)} labels "
                       f"(highest: {top_label} {top_prob:.2f}, below the 0.5 operating point).")
            findings.append(
                FindingItem(
                    name="CXR: No label above operating point",
                    status=FindingStatus.ABSENT,
                    confidence=round(1.0 - top_prob, 3),
                    details={"highest_label": top_label, "highest_score": top_prob},
                )
            )


        return ToolResult(
            tool_name=self.name,
            modality=self.modality,
            findings=findings,
            summary=summary,
            raw_output={"probabilities": probabilities, "image_path": image_path},
        )
