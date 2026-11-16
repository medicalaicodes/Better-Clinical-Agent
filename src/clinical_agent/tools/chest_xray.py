"""
Chest X-ray tool: a DenseNet-121 from TorchXRayVision (Cohen et al., 2022), trained on several
large public chest X-ray datasets. It scores the image for up to 18 findings (consolidation,
effusion, cardiomegaly, ...).

Its scores are already calibrated: for each finding, 0.5 is the threshold the library's authors
picked on validation data. So a score of 0.5 or more means "positive". (The old code applied a
sigmoid a second time, which squashed every score into 0.50-0.73 - fixed in Phase 1.)
"""

import math
import os

from ..schemas import FindingItem, FindingStatus, ModalityType, ToolResult
from .base import BaseClinicalTool


class ChestXRayClassifierTool(BaseClinicalTool):
    name = "chest_xray_classifier"
    description = ("Chest X-ray classifier (DenseNet-121): scores the radiograph for 18 chest findings "
                   "such as consolidation, lung opacity, effusion, cardiomegaly, pneumothorax and edema.")
    modality = ModalityType.IMAGING

    def __init__(self, device=None):
        self.device = device
        self.model = None             # loaded the first time an image is classified
        self.labels = []

    def load_model(self):
        import torch
        import torchxrayvision as xrv

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = xrv.models.DenseNet(weights="densenet121-res224-all").to(self.device).eval()

        # Some labels were never trained in this checkpoint: their threshold is NaN and their
        # output is a meaningless constant, so we keep only the labels with a real threshold.
        self.labels = []
        thresholds = self.model.op_threshs
        for i in range(len(self.model.pathologies)):
            label = self.model.pathologies[i]
            if label and not math.isnan(float(thresholds[i])):
                self.labels.append(label)

    def classify(self, image_path):
        """{finding: score between 0 and 1} for one image."""
        import skimage.io
        import torch
        import torchxrayvision as xrv

        if self.model is None:
            self.load_model()

        image = skimage.io.imread(image_path)
        image = xrv.datasets.normalize(image, 255)        # pixel values into the range the model expects
        if len(image.shape) > 2:
            image = image.mean(axis=2)                     # colour -> greyscale
        image = image[None, ...]                           # add the "channel" dimension
        image = xrv.datasets.XRayCenterCrop()(image)
        image = xrv.datasets.XRayResizer(224)(image)       # the model was trained on 224x224 images

        tensor = torch.from_numpy(image).unsqueeze(0).to(self.device).float()
        with torch.no_grad():
            outputs = self.model(tensor).cpu().numpy()[0]

        scores = {}
        for label, score in zip(self.model.pathologies, outputs):
            if label in self.labels:
                scores[label] = round(float(score), 4)
        return scores

    def run(self, case, **options):
        image_path = options.get("image_path") or case.image_path
        if not image_path or not os.path.exists(image_path):
            return ToolResult(tool_name=self.name, modality=self.modality, findings=[],
                              summary="No chest radiograph available for this patient.",
                              raw_output={"status": "image_not_found"})

        scores = self.classify(image_path)

        # Highest score first.
        ranked = []
        for label, score in scores.items():
            ranked.append((score, label))
        ranked.sort(reverse=True)

        findings = []
        positives = []
        for score, label in ranked:
            if score >= 0.5:
                positives.append(f"{label} ({score:.2f})")
                findings.append(FindingItem(name=f"CXR: {label}", confidence=score, quantitative_value=score,
                                            unit="calibrated score (0.5 = threshold)"))

        if positives:
            summary = f"Chest radiograph classifier is positive (score >= 0.5) for: {', '.join(positives[:4])}."
        else:
            top_score, top_label = ranked[0]
            summary = (f"Chest radiograph classifier is negative for all {len(scores)} findings "
                       f"(highest: {top_label} {top_score:.2f}, below 0.5).")
            findings.append(FindingItem(name="CXR: no finding above threshold", status=FindingStatus.ABSENT,
                                        confidence=round(1.0 - top_score, 3)))

        return ToolResult(tool_name=self.name, modality=self.modality, findings=findings, summary=summary,
                          raw_output={"probabilities": scores, "image_path": image_path})
