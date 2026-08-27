"""
Clinical AI Agent: Evaluation Metrics Suite
Computes Top-K Diagnostic Accuracy, Evidence Grounding Rate, and Clinical Agreement.

Academic Citations & Foundations:
- SDBench: "A Benchmark for Sequential Diagnosis with Language Models" (Microsoft Research, 2025; arXiv:2506.22405).
- MedAgent-Pro: "Towards Evidence-Based Multi-Modal Medical Diagnosis via Reasoning Agentic Workflow" (Wang et al., ICLR 2026; arXiv:2503.18968).
"""

from typing import List, Dict, Any, Optional
from ..schemas import ClinicalReport


def compute_top_k_accuracy(
    predictions: List[List[str]],
    ground_truths: List[str],
    k: int = 1,
) -> float:
    """
    Calculates Top-K diagnostic accuracy.
    Checks if ground truth disease appears in the top-K ranked differential.
    """
    if not ground_truths or not predictions:
        return 0.0

    correct = 0
    total = len(ground_truths)

    for pred_list, gt in zip(predictions, ground_truths):
        gt_lower = gt.lower().strip()
        gt_aliases = [gt_lower] + [a.strip() for a in gt_lower.replace("(", "/").replace(")", "/").split("/") if len(a.strip()) > 3]
        top_k_preds = [p.lower().strip() for p in pred_list[:k]]
        
        is_match = False
        for alias in gt_aliases:
            if any(alias in p or p in alias for p in top_k_preds):
                is_match = True
                break
        
        if is_match:
            correct += 1

    return round(correct / total, 4)


def compute_grounding_rate(reports: List[ClinicalReport]) -> float:
    """
    Computes Evidence Grounding Rate:
    Percentage of finalized consultation reports that successfully gathered structured evidence packets.
    """
    if not reports:
        return 0.0

    grounded = 0
    for r in reports:
        if len(r.evidence_trail) > 0 and len(r.key_findings) > 0:
            grounded += 1

    return round(grounded / len(reports), 4)

