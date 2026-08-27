"""
Clinical AI Agent: TorchXRayVision DenseNet-121 Model Wrapper

Academic Citations & Foundations:
- TorchXRayVision Model Zoo:
  * Cohen et al., "TorchXRayVision: A library of chest X-ray datasets and models" (Medical Imaging with Deep Learning - MIDL 2022; arXiv:2111.00595).
  * Checkpoint: densenet121-res224-all (NIH ChestX-ray14, PadChest, CheXpert, MIMIC-CXR, Google, OpenI, Kaggle).
- Dense Convolutional Networks:
  * Huang et al., "Densely Connected Convolutional Networks" (CVPR 2017; arXiv:1608.06993).
"""

import os
import threading
from typing import Dict, List, Optional
import numpy as np
import torch
from PIL import Image

try:
    import torchxrayvision as xrv
    import skimage.io
except ImportError:
    xrv = None


class DenseNetCXRModel:
    """Thread-safe Singleton wrapper for TorchXRayVision DenseNet-121."""
    _instance: Optional["DenseNetCXRModel"] = None
    _lock = threading.Lock()

    def __new__(cls, weights: str = "densenet121-res224-all", device: Optional[str] = None):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(DenseNetCXRModel, cls).__new__(cls)
                cls._instance._initialized = False
        return cls._instance

    def __init__(self, weights: str = "densenet121-res224-all", device: Optional[str] = None):
        if self._initialized:
            return
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.weights_name = weights
        self.model = None
        self.pathologies = []
        self.op_threshs: Dict[str, float] = {}
        self._load_model()
        self._initialized = True

    def _load_model(self):
        if xrv is None:
            raise ImportError("torchxrayvision or scikit-image is not installed. Please install requirements.")
        try:
            self.model = xrv.models.DenseNet(weights=self.weights_name).to(self.device)
            self.model.eval()
            self.pathologies = list(self.model.pathologies)
            if hasattr(self.model, "op_threshs") and self.model.op_threshs is not None:
                for idx, p in enumerate(self.pathologies):
                    if idx < len(self.model.op_threshs):
                        val = self.model.op_threshs[idx]
                        if not np.isnan(val):
                            self.op_threshs[p] = float(val)
                        else:
                            self.op_threshs[p] = 0.50
            else:
                for p in self.pathologies:
                    self.op_threshs[p] = 0.50
        except Exception as e:
            self.model = None
            raise RuntimeError(f"Failed to load TorchXRayVision DenseNet model '{self.weights_name}': {e}")


    def predict(self, image_path: str) -> Dict[str, float]:
        """Runs forward inference on a chest X-ray image and returns calibrated pathology probabilities."""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at path: {image_path}")

        if self.model is None or xrv is None:
            raise RuntimeError("DenseNet model could not be initialized.")

        img = skimage.io.imread(image_path)
        img = xrv.datasets.normalize(img, 255)

        if len(img.shape) > 2:
            img = img.mean(axis=2)
        img = img[None, ...]

        transform = xrv.datasets.XRayCenterCrop()
        img = transform(img)
        transform_rescale = xrv.datasets.XRayResizer(224)
        img = transform_rescale(img)

        tensor = torch.from_numpy(img).unsqueeze(0).to(self.device).float()
        with torch.no_grad():
            preds = self.model(tensor).cpu().numpy()[0]

        results = {}
        for path_name, score in zip(self.pathologies, preds):
            prob = 1.0 / (1.0 + np.exp(-float(score)))
            results[path_name] = round(float(prob), 4)

        return results
