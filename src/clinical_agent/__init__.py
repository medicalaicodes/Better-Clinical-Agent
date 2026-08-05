"""
Clinical Agent Framework
Evidence-Reliable Multimodal AI Reasoning for Healthcare
"""

__version__ = "0.1.0"

from .schemas import (
    ModalityType,
    EvidencePacket,
    AgentClaim,
    ConflictLog,
    ClinicalReport,
)
from .router import SpecialistRouter
from .coordination import EvidenceCoordinator

__all__ = [
    "ModalityType",
    "EvidencePacket",
    "AgentClaim",
    "ConflictLog",
    "ClinicalReport",
    "SpecialistRouter",
    "EvidenceCoordinator",
]
