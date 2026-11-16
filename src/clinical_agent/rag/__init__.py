"""Medical question answering: the published baselines (methods.py) and our diagnostic agent (agent.py)."""

from .agent import DxAgent
from .methods import MCQ, METHODS, RETRIEVAL_FREE

METHODS["dx_agent"] = DxAgent

__all__ = ["MCQ", "METHODS", "RETRIEVAL_FREE", "DxAgent"]
