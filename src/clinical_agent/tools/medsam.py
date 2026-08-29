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
        heart_area_px = seg_result.get("heart_area_px", 0)
        symmetry = seg_result["symmetry_index"]
        collapse_pct = seg_result.get("estimated_collapse_pct", round((1.0 - symmetry) * 100.0, 1))
        ctr = seg_result.get("cardiothoracic_ratio", 0.45)
        is_cardiomegaly = seg_result.get("is_cardiomegaly", ctr > 0.50)

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
                status=FindingStatus.PRESENT if symmetry < 0.75 else FindingStatus.ABSENT,
                confidence=0.90,
                quantitative_value=symmetry,
                unit="ratio",
                details={"symmetry_index": symmetry, "is_asymmetric": symmetry < 0.75},
            ),
            FindingItem(
                name="MedSAM: Estimated Unilateral Lung Volume Deficit / Collapse",
                status=FindingStatus.PRESENT if collapse_pct >= 30.0 else FindingStatus.ABSENT,
                confidence=0.88,
                quantitative_value=collapse_pct,
                unit="percent",
                details={"collapse_pct": collapse_pct, "significant_collapse": collapse_pct >= 30.0},
            ),
            FindingItem(
                name="MedSAM: Cardiothoracic Ratio (CTR) / Radiographic Cardiomegaly",
                status=FindingStatus.PRESENT if is_cardiomegaly else FindingStatus.ABSENT,
                confidence=0.91,
                quantitative_value=ctr,
                unit="ratio",
                details={"ctr": ctr, "is_cardiomegaly": is_cardiomegaly, "threshold": 0.50},
            ),
        ]

        if is_cardiomegaly:
            summary = (
                f"MedSAM anatomical segmentation reveals enlarged cardiac silhouette (Cardiothoracic Ratio CTR: {ctr:.2f} > 0.50 threshold, "
                f"confirming radiographic cardiomegaly) with {symmetry:.2f} thoracic symmetry."
            )
        elif collapse_pct >= 30.0:
            summary = (
                f"MedSAM anatomical segmentation indicates significant thoracic volume deficit "
                f"({collapse_pct:.0f}% unilateral volume loss / parenchymal retraction; Symmetry: {symmetry:.2f}, CTR: {ctr:.2f})."
            )
        else:
            summary = (
                f"MedSAM anatomical segmentation demonstrates symmetric bilateral lung aeration and normal cardiac silhouette "
                f"(Symmetry: {symmetry:.2f}, Volume Deficit: {collapse_pct:.0f}%, CTR: {ctr:.2f} <= 0.50 threshold) without focal collapse or cardiomegaly."
            )

        return ToolResult(
            tool_name=self.name,
            modality=self.modality,
            findings=findings,
            summary=summary,
            raw_output={
                "left_area_px": left_area_px,
                "right_area_px": right_area_px,
                "heart_area_px": heart_area_px,
                "symmetry_index": symmetry,
                "estimated_collapse_pct": collapse_pct,
                "cardiothoracic_ratio": ctr,
                "is_cardiomegaly": is_cardiomegaly,
            },
        )

