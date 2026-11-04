"""
Medical knowledge tool: the agent's door into the retrieval engine.

It replaces the old 9-document TF-IDF "guideline index". Queries go through the same hybrid
retriever (BM25 + MedCPT + cross-encoder) over Textbooks + StatPearls that the MIRAGE
benchmark evaluates, so the agent uses a retriever with measured quality.
"""

import os
from typing import List, Optional

from ..config import get_settings
from ..retrieval import Query, Retriever
from ..schemas import FindingItem, FindingStatus, ModalityType, PatientCase, ToolResult
from .base import BaseClinicalTool


class MedicalKnowledgeRAGTool(BaseClinicalTool):
    name: str = "medical_knowledge_rag"
    description: str = (
        "Searches medical textbooks and StatPearls (~427k passages) with hybrid keyword + semantic "
        "retrieval and returns the most relevant cited passages for a clinical query."
    )
    modality: ModalityType = ModalityType.KNOWLEDGE

    def __init__(self, retriever: Optional[Retriever] = None, k: int = 4):
        self._retriever = retriever       # loaded lazily: the index is large
        self.k = k

    def _get_retriever(self) -> Optional[Retriever]:
        if self._retriever is None:
            index_dir = get_settings().index_dir
            if not os.path.exists(os.path.join(index_dir, "corpus.jsonl")):
                return None
            self._retriever = Retriever.load(index_dir, device=get_settings().device)
        return self._retriever

    def run(self, case: PatientCase, **kwargs) -> ToolResult:
        query = kwargs.get("query") or case.chief_complaint
        retriever = self._get_retriever()
        if retriever is None:
            return ToolResult(tool_name=self.name, modality=self.modality, findings=[],
                              summary="Knowledge index not built (run scripts/build_index.py).",
                              raw_output={"status": "no_index"})

        evidence = retriever.search_many([Query(query, "AGENT")], budget=self.k, per_query=self.k)
        findings: List[FindingItem] = [
            FindingItem(
                name=f"Reference [{e.source}: {e.title}]",
                status=FindingStatus.PRESENT,
                confidence=1.0,
                details={"snippet_id": e.snippet.id, "text": e.text, "rerank_score": round(e.score, 3)},
            )
            for e in evidence
        ]
        summary = "\n".join(f"[{e.snippet.id}] ({e.source}: {e.title}) {e.text[:400]}" for e in evidence)
        return ToolResult(tool_name=self.name, modality=self.modality, findings=findings,
                          summary=f"Retrieved {len(evidence)} passages for '{query}':\n{summary}",
                          raw_output={"query": query, "snippet_ids": [e.snippet.id for e in evidence]})
