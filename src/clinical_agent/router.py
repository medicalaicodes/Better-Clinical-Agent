"""
Specialist Router
Dispatches patient case inputs to relevant modality specialists and detects missing modalities.
"""

from typing import List, Dict, Set, Any
from .schemas import ModalityType


class SpecialistRouter:
    """
    Routes multimodal patient data to specialized agents (Imaging, EHR, Notes)
    and tracks missing inputs without failing.
    """
    def __init__(self, required_modalities: List[ModalityType] = None):
        self.required_modalities = required_modalities or [
            ModalityType.CXR,
            ModalityType.EHR,
            ModalityType.NOTES,
        ]

    def route_inputs(self, provided_data: Dict[ModalityType, Any]) -> Dict[str, Any]:
        """
        Identifies available modalities and logs missing modalities.
        """
        available: Set[ModalityType] = set(provided_data.keys())
        missing: List[ModalityType] = [
            mod for mod in self.required_modalities if mod not in available
        ]
        
        return {
            "available_modalities": list(available),
            "missing_modalities": missing,
            "data_payloads": provided_data,
        }
