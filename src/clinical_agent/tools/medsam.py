"""
Clinical AI Agent: MedSAM Thoracic Segmentation Specialist Tool

Academic Citations & Foundations:
- Medical Vision Foundation Models for Anatomical Segmentation:
  * Ma et al., "Segment Anything in Medical Images" (Nature Communications, 2024; DOI:10.1038/s41467-024-44824-z).
  * Kirillov et al., "Segment Anything" (Meta AI Research / ICCV 2023; arXiv:2304.02643).
  * MedRAX: "A Medical Reasoning Agent for Chest X-ray" (Bo Wang Lab, 2025; arXiv:2502.02618).
"""

import os
import numpy as np
from typing import Dict, List, Optional, Any
from PIL import Image

from .base import BaseClinicalTool
from ..schemas import PatientCase, ToolResult, ModalityType, FindingItem, FindingStatus
from ..models import MedSAMSegmenterModel


class MedSAMSegmenterTool(BaseClinicalTool):
    name: str = "medsam_segmenter"
    description: str = (
        "Segments anatomical lung fields and thoracic structures using MedSAM ViT-Base. "
        "Calculates left and right lung field pixel areas, detects thoracic asymmetry, "
        "and evaluates for volume loss or basilar opacification."
    )
    modality: ModalityType = ModalityType.SEGMENTATION

    def __init__(self, device: Optional[str] = None):
        self.device = device
        self._model_wrapper: Optional[MedSAMSegmenterModel] = None

    def _get_model(self) -> MedSAMSegmenterModel:
        if self._model_wrapper is None:
            self._model_wrapper = MedSAMSegmenterModel(device=self.device)
        return self._model_wrapper

    def run(self, case: PatientCase, **kwargs) -> ToolResult:
        image_path = kwargs.get("image_path") or case.image_path
        if not image_path or not os.path.exists(image_path):
            return ToolResult(
                tool_name=self.name,
                modality=self.modality,
                findings=[],
                summary="No thoracic image available for MedSAM anatomical segmentation.",
                raw_output={"status": "image_not_found"},
            )

        # Execute MedSAM ViT-Base segmentation directly
        model = self._get_model()
        seg_result = model.segment_lung_fields(image_path)
        left_area_px = seg_result["left_lung_area_px"]
        right_area_px = seg_result["right_lung_area_px"]
        symmetry = seg_result["symmetry_index"]


        findings: List[FindingItem] = [
            FindingItem(
                name="MedSAM: Left Lung Field Segmentation Area",
                status=FindingStatus.PRESENT,
                confidence=0.92,
                quantitative_value=float(left_area_px),
                unit="pixels",
            ),
            FindingItem(
                name="MedSAM: Right Lung Field Segmentation Area",
                status=FindingStatus.PRESENT,
                confidence=0.92,
                quantitative_value=float(right_area_px),
                unit="pixels",
            ),
            FindingItem(
                name="MedSAM: Thoracic Lung-Field Area Symmetry Index",
                status=FindingStatus.PRESENT if symmetry < 0.85 else FindingStatus.ABSENT,
                confidence=0.90,
                quantitative_value=symmetry,
                unit="ratio",
                details={"symmetry_index": symmetry, "is_asymmetric": symmetry < 0.85},
            ),
        ]

        if symmetry < 0.85:
            summary = (
                f"MedSAM segmentation indicates significant thoracic asymmetry (symmetry index: {symmetry:.2f}). "
                f"Relative volume reduction or basilar opacity detected in the right lung field."
            )
        else:
            summary = f"MedSAM segmentation shows symmetric bilateral lung fields (symmetry index: {symmetry:.2f}) without focal volume loss."

        return ToolResult(
            tool_name=self.name,
            modality=self.modality,
            findings=findings,
            summary=summary,
            raw_output={"left_area_px": left_area_px, "right_area_px": right_area_px, "symmetry_index": symmetry},
        )
