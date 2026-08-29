"""
Clinical AI Agent: MedSAM (Medical Segment Anything) Model Wrapper

Academic Citations & Foundations:
- Medical Image Segmentation with Foundation Models:
  * Ma et al., "Segment Anything in Medical Images" (Nature Communications, 2024; DOI:10.1038/s41467-024-44824-z).
  * HuggingFace Model Hub: wanglab/medsam-vit-base (Bo Wang Lab, University of Toronto / UHN).
- Segment Anything Architecture:
  * Kirillov et al., "Segment Anything" (Meta AI Research / ICCV 2023; arXiv:2304.02643).
"""

import os
import threading
from typing import Dict, Optional, Tuple, Any
import numpy as np
import torch
from PIL import Image

try:
    from transformers import AutoProcessor, AutoModelForMaskGeneration
except ImportError:
    AutoProcessor = None
    AutoModelForMaskGeneration = None


class MedSAMSegmenterModel:
    """Thread-safe Singleton wrapper for MedSAM (wanglab/medsam-vit-base)."""
    _instance: Optional["MedSAMSegmenterModel"] = None
    _lock = threading.Lock()

    def __new__(cls, model_id: str = "wanglab/medsam-vit-base", device: Optional[str] = None):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(MedSAMSegmenterModel, cls).__new__(cls)
                cls._instance._initialized = False
        return cls._instance

    def __init__(self, model_id: str = "wanglab/medsam-vit-base", device: Optional[str] = None):
        if self._initialized:
            return
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_id = model_id
        self.processor = None
        self.model = None
        self._load_model()
        self._initialized = True

    def _load_model(self):
        if AutoProcessor is None or AutoModelForMaskGeneration is None:
            raise ImportError("transformers or torch is not installed. Please install requirements.")
        try:
            self.processor = AutoProcessor.from_pretrained(self.model_id)
            self.model = AutoModelForMaskGeneration.from_pretrained(self.model_id).to(self.device)
            self.model.eval()
        except Exception as e:
            self.processor = None
            self.model = None
            raise RuntimeError(f"Failed to load MedSAM model '{self.model_id}': {e}")

    def segment_lung_fields(self, image_path: str) -> Dict[str, Any]:
        """Performs prompt-guided thoracic segmentation for left/right lung fields and cardiac silhouette."""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at path: {image_path}")

        if self.model is None or self.processor is None:
            raise RuntimeError("MedSAM model is not initialized.")

        image = Image.open(image_path).convert("RGB")
        w, h = image.size

        # 3 Bounding boxes: Right Lung (viewer left), Left Lung (viewer right), and Cardiac Silhouette
        input_boxes = [[[
            int(w * 0.04), int(h * 0.12), int(w * 0.48), int(h * 0.88)
        ], [
            int(w * 0.52), int(h * 0.12), int(w * 0.96), int(h * 0.88)
        ], [
            int(w * 0.25), int(h * 0.40), int(w * 0.75), int(h * 0.88)
        ]]]

        try:
            inputs = self.processor(image, input_boxes=input_boxes, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)

            masks = self.processor.image_processor.post_process_masks(
                outputs.pred_masks.cpu(),
                inputs["original_sizes"].cpu(),
                inputs["reshaped_input_sizes"].cpu(),
            )

            # Extract distinct masks for each prompt box
            right_mask = masks[0][0][0].numpy() > 0
            left_mask = masks[0][1][0].numpy() > 0
            heart_mask = masks[0][2][0].numpy() > 0

            right_lung_area = int(np.sum(right_mask))
            left_lung_area = int(np.sum(left_mask))
            heart_area = int(np.sum(heart_mask))

            # Transverse diameters for Cardiothoracic Ratio (CTR)
            heart_cols = np.where(np.any(heart_mask, axis=0))[0]
            heart_width = (heart_cols[-1] - heart_cols[0]) if len(heart_cols) > 0 else int(w * 0.40)

            lung_cols = np.where(np.any(right_mask | left_mask, axis=0))[0]
            thorax_width = (lung_cols[-1] - lung_cols[0]) if len(lung_cols) > 0 else int(w * 0.85)

            ctr = round(float(heart_width / max(thorax_width, 1)), 2)
            if left_lung_area > 0 and right_lung_area > 0:
                symmetry = round(min(left_lung_area, right_lung_area) / max(left_lung_area, right_lung_area), 2)
            else:
                symmetry = 1.00

            collapse_pct = round((1.0 - symmetry) * 100.0, 1)

            return {
                "left_lung_area_px": left_lung_area,
                "right_lung_area_px": right_lung_area,
                "heart_area_px": heart_area,
                "symmetry_index": symmetry,
                "estimated_collapse_pct": collapse_pct,
                "cardiothoracic_ratio": ctr,
                "is_cardiomegaly": ctr > 0.50,
            }
        except Exception as e:
            raise RuntimeError(f"MedSAM inference failed on '{image_path}': {e}")


