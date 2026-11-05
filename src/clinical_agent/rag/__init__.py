"""Medical question answering: baselines from the literature, DDx-RAG (v1) and DxAgent (ours)."""

from .agent import DxAgent
from .methods import METHODS, MCQ, RETRIEVAL_FREE, DDxRAG, format_evidence

METHODS[DxAgent.name] = DxAgent

__all__ = ["METHODS", "MCQ", "RETRIEVAL_FREE", "DDxRAG", "DxAgent", "format_evidence"]
