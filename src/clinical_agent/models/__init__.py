"""
Clinical AI Agent: Neural Vision & Foundation Models Module
Centralizes model lifecycles, memory caching, and device management for vision and reasoning backends.
"""

from .densenet_cxr import DenseNetCXRModel
from .medsam_wrapper import MedSAMSegmenterModel


def ensure_models_downloaded(verbose: bool = True):
    """
    Ensures all pretrained foundation neural weights are downloaded and cached:
    1. TorchXRayVision DenseNet-121 ('densenet121-res224-all')
    2. MedSAM ViT-Base ('wanglab/medsam-vit-base')
    """
    if verbose:
        print("=" * 80)
        print("  VERIFYING FOUNDATION VISION MODEL CHECKPOINTS (INITIAL RUN)")
        print("=" * 80)

    # 1. DenseNet-121 Checkpoint
    if verbose:
        print("[1/2] Checking / Downloading TorchXRayVision DenseNet-121 weights ('densenet121-res224-all')...")
    DenseNetCXRModel()
    if verbose:
        print("      DenseNet-121 checkpoint verified.")

    # 2. MedSAM ViT-Base Checkpoint
    if verbose:
        print("[2/2] Checking / Downloading MedSAM ViT-Base weights ('wanglab/medsam-vit-base')...")
    MedSAMSegmenterModel()
    if verbose:
        print("      MedSAM ViT-Base checkpoint verified.")
        print("=" * 80)
        print("  ALL NEURAL FOUNDATION MODELS LOADED & READY\n")


__all__ = [
    "DenseNetCXRModel",
    "MedSAMSegmenterModel",
    "ensure_models_downloaded",
]
