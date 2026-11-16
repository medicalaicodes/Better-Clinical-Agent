"""
Chest segmentation tool: MedSAM (Ma et al., Nature Communications 2024), the "Segment Anything"
model fine-tuned on medical images. We give it three boxes on the X-ray - right lung, left lung,
heart - and it outlines each. From the outlines we measure:

    lung symmetry     smaller lung area / bigger lung area (1.0 = equal). Low = one side collapsed.
    collapse %        how much smaller the smaller lung is
    CTR               cardiothoracic ratio = heart width / chest width. Above 0.50 = enlarged heart.
"""

import os

from ..schemas import FindingItem, FindingStatus, ModalityType, ToolResult
from .base import BaseClinicalTool


class MedSAMSegmenterTool(BaseClinicalTool):
    name = "medsam_segmenter"
    description = ("Chest segmenter (MedSAM): outlines both lungs and the heart on the radiograph and "
                   "measures lung symmetry, lung collapse and the cardiothoracic ratio (heart size).")
    modality = ModalityType.SEGMENTATION

    def __init__(self, device=None):
        self.device = device
        self.model = None              # loaded the first time an image is segmented
        self.processor = None

    def load_model(self):
        import torch
        from transformers import AutoModelForMaskGeneration, AutoProcessor

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained("wanglab/medsam-vit-base")
        self.model = AutoModelForMaskGeneration.from_pretrained("wanglab/medsam-vit-base").to(self.device).eval()

    def measure(self, image_path):
        import numpy as np
        import torch
        from PIL import Image

        if self.model is None:
            self.load_model()

        image = Image.open(image_path).convert("RGB")
        width, height = image.size

        # Three boxes as fractions of the image (left, top, right, bottom). On an X-ray the
        # patient's RIGHT lung is on the viewer's LEFT.
        right_lung_box = [int(width * 0.04), int(height * 0.12), int(width * 0.48), int(height * 0.88)]
        left_lung_box = [int(width * 0.52), int(height * 0.12), int(width * 0.96), int(height * 0.88)]
        heart_box = [int(width * 0.25), int(height * 0.40), int(width * 0.75), int(height * 0.88)]

        inputs = self.processor(image, input_boxes=[[right_lung_box, left_lung_box, heart_box]],
                                return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        masks = self.processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu())

        # One yes/no pixel mask per box.
        right_mask = masks[0][0][0].numpy() > 0
        left_mask = masks[0][1][0].numpy() > 0
        heart_mask = masks[0][2][0].numpy() > 0

        right_area = int(np.sum(right_mask))
        left_area = int(np.sum(left_mask))

        # Width = distance between the leftmost and rightmost columns that contain the shape.
        heart_columns = np.where(np.any(heart_mask, axis=0))[0]
        if len(heart_columns) > 0:
            heart_width = heart_columns[-1] - heart_columns[0]
        else:
            heart_width = int(width * 0.40)

        lung_columns = np.where(np.any(right_mask | left_mask, axis=0))[0]
        if len(lung_columns) > 0:
            chest_width = lung_columns[-1] - lung_columns[0]
        else:
            chest_width = int(width * 0.85)

        ctr = round(float(heart_width / max(chest_width, 1)), 2)
        if left_area > 0 and right_area > 0:
            symmetry = round(min(left_area, right_area) / max(left_area, right_area), 2)
        else:
            symmetry = 1.0

        return {"left_lung_area_px": left_area, "right_lung_area_px": right_area, "symmetry_index": symmetry,
                "estimated_collapse_pct": round((1.0 - symmetry) * 100.0, 1), "cardiothoracic_ratio": ctr}

    def run(self, case, **options):
        image_path = options.get("image_path") or case.image_path
        if not image_path or not os.path.exists(image_path):
            return ToolResult(tool_name=self.name, modality=self.modality, findings=[],
                              summary="No chest image available for segmentation.",
                              raw_output={"status": "image_not_found"})

        m = self.measure(image_path)
        symmetry = m["symmetry_index"]
        collapse = m["estimated_collapse_pct"]
        ctr = m["cardiothoracic_ratio"]
        enlarged_heart = ctr > 0.50

        def present_if(condition):
            if condition:
                return FindingStatus.PRESENT
            return FindingStatus.ABSENT

        findings = [
            FindingItem(name="MedSAM: Left Lung Field Segmentation Area", confidence=0.92,
                        quantitative_value=float(m["left_lung_area_px"]), unit="pixels"),
            FindingItem(name="MedSAM: Right Lung Field Segmentation Area", confidence=0.92,
                        quantitative_value=float(m["right_lung_area_px"]), unit="pixels"),
            FindingItem(name="MedSAM: Lung Symmetry Index", status=present_if(symmetry < 0.75), confidence=0.90,
                        quantitative_value=symmetry, unit="ratio"),
            FindingItem(name="MedSAM: Lung Collapse", status=present_if(collapse >= 30.0), confidence=0.88,
                        quantitative_value=collapse, unit="percent"),
            FindingItem(name="MedSAM: Cardiothoracic Ratio (Cardiomegaly)", status=present_if(enlarged_heart),
                        confidence=0.91, quantitative_value=ctr, unit="ratio"),
        ]

        if enlarged_heart:
            summary = (f"Segmentation shows an enlarged heart (cardiothoracic ratio {ctr:.2f}, above 0.50) "
                       f"with lung symmetry {symmetry:.2f}.")
        elif collapse >= 30.0:
            summary = (f"Segmentation shows significant volume loss on one side ({collapse:.0f}% smaller; "
                       f"symmetry {symmetry:.2f}, cardiothoracic ratio {ctr:.2f}).")
        else:
            summary = (f"Segmentation shows symmetric lungs and a normal heart size "
                       f"(symmetry {symmetry:.2f}, volume difference {collapse:.0f}%, cardiothoracic ratio {ctr:.2f}).")

        return ToolResult(tool_name=self.name, modality=self.modality, findings=findings, summary=summary,
                          raw_output=m)
