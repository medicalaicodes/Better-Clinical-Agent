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

        # Extract structured findings with explicit classification margins
        findings: List[FindingItem] = []
        for path_name, prob in sorted(probabilities.items(), key=lambda x: x[1], reverse=True):
            thresh = thresholds.get(path_name, 0.50)
            is_present = prob >= thresh
            margin = round(abs(prob - thresh), 4)

            # Record all present pathologies or high-probability candidates
            if is_present or prob >= 0.40:
                findings.append(
                    FindingItem(
                        name=f"CXR: {path_name}",
                        status=FindingStatus.PRESENT if is_present else FindingStatus.ABSENT,
                        confidence=prob,
                        quantitative_value=prob,
                        unit="calibrated_probability",
                        details={
                            "operating_threshold": thresh,
                            "classification_margin": margin,
                            "is_positive": is_present,
                        },
                    )
                )

        present_names = [f.name.replace("CXR: ", "") for f in findings if f.status == FindingStatus.PRESENT]
        if present_names:
            top_pathology = present_names[0]
            top_prob = probabilities.get(top_pathology, 0.5)
            thresh = thresholds.get(top_pathology, 0.50)
            margin = round(abs(top_prob - thresh), 4)
            summary = (
                f"Chest radiograph reveals {', '.join(present_names[:3])} "
                f"(Lead: {top_pathology}, Prob: {top_prob:.3f}, Threshold: {thresh:.2f}, Margin: {margin:.3f})."
            )
        else:
            summary = "Chest radiograph shows no acute cardiopulmonary consolidation, effusion, or pneumothorax."
            findings.append(
                FindingItem(
                    name="CXR: Clear Lungs / No Acute Cardiopulmonary Abnormality",
                    status=FindingStatus.PRESENT,
                    confidence=0.95,
                )
            )

        return ToolResult(
            tool_name=self.name,
            modality=self.modality,
            findings=findings,
            summary=summary,
            raw_output={"probabilities": probabilities, "image_path": image_path},
        )
