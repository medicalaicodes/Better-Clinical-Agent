"""
Medical library tool: the clinical agent's door into the retrieval engine (retrieval/search.py).
It searches the same 427k textbook and StatPearls passages, with the same hybrid search, that we
benchmarked on MIRAGE - so the agent uses a retriever whose quality we have actually measured.
"""

import os

from ..config import get_settings
from ..retrieval import load_retriever
from ..schemas import FindingItem, ModalityType, ToolResult
from .base import BaseClinicalTool


class MedicalKnowledgeRAGTool(BaseClinicalTool):
    name = "medical_knowledge_rag"
    description = ("Medical library search: finds the most relevant passages in medical textbooks and "
                   "StatPearls (~427k passages) for a short clinical query.")
    modality = ModalityType.KNOWLEDGE

    def __init__(self, retriever=None, k=4):
        self.retriever = retriever       # loaded the first time it's needed: the index is big
        self.k = k

    def get_retriever(self):
        if self.retriever is None:
            index_dir = get_settings().index_dir
            if not os.path.exists(os.path.join(index_dir, "corpus.jsonl")):
                return None
            self.retriever = load_retriever(index_dir, device=get_settings().device)
        return self.retriever

    def run(self, case, **options):
        query = options.get("query") or case.chief_complaint
        retriever = self.get_retriever()
        if retriever is None:
            return ToolResult(tool_name=self.name, modality=self.modality, findings=[],
                              summary="Knowledge index not built (run scripts/build_index.py).",
                              raw_output={"status": "no_index"})

        evidence = retriever.search(query, k=self.k)

        findings = []
        summary_lines = []
        snippet_ids = []
        for item in evidence:
            findings.append(FindingItem(
                name=f"Reference [{item.source}: {item.title}]",
                confidence=1.0,
                details={"snippet_id": item.snippet.id, "text": item.text, "rerank_score": round(item.score, 3)},
            ))
            summary_lines.append(f"[{item.snippet.id}] ({item.source}: {item.title}) {item.text[:400]}")
            snippet_ids.append(item.snippet.id)

        summary = f"Retrieved {len(evidence)} passages for '{query}':\n" + "\n".join(summary_lines)
        return ToolResult(tool_name=self.name, modality=self.modality, findings=findings,
                          summary=summary, raw_output={"query": query, "snippet_ids": snippet_ids})
