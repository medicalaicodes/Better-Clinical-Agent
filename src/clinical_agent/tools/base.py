"""
Clinical AI Agent: Base Specialist Tool Interface

Academic Citations & Foundations:
- Tool-Augmented Language Models & Modular Tool Frameworks:
  * Schick et al., "Toolformer: Language Models Can Teach Themselves to Use Tools" (NeurIPS 2023; arXiv:2302.04761).
  * MedRAX: "A Medical Reasoning Agent for Chest X-ray" (Bo Wang Lab, 2025; arXiv:2502.02618) - BaseTool modularity.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from pydantic import BaseModel

from ..schemas import PatientCase, ToolResult, ModalityType


class BaseClinicalTool(ABC):
    """Abstract base class for modular clinical specialist tools."""
    name: str = "base_tool"
    description: str = "Base clinical specialist tool"
    modality: ModalityType = ModalityType.CLINICAL_NOTES

    @abstractmethod
    def run(self, case: PatientCase, **kwargs) -> ToolResult:
        """
        Executes the tool on the given patient case and returns a standardized ToolResult.
        """
        pass

    def __call__(self, case: PatientCase, **kwargs) -> ToolResult:
        return self.run(case, **kwargs)
