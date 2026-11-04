"""Medical question answering: baselines from the literature and DDx-RAG (ours)."""

from .methods import METHODS, MCQ, RETRIEVAL_FREE, DDxRAG, format_evidence

__all__ = ["METHODS", "MCQ", "RETRIEVAL_FREE", "DDxRAG", "format_evidence"]
