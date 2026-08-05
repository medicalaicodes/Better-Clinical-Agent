"""
Evidence Coordinator
Merges claims from specialists, checks for cross-modal conflict, and preserves disagreement signal.
"""

from typing import List
from .schemas import (
    ModalityType,
    AgentClaim,
    ConflictLog,
    ClinicalReport,
)


class EvidenceCoordinator:
    """
    Coordinates multi-specialist claims, calculates epistemic calibration,
    and constructs the final clinical report while explicitly preserving conflicts.
    """
    def __init__(self, conflict_threshold: float = 0.5):
        self.conflict_threshold = conflict_threshold

    def synthesize(
        self,
        claims: List[AgentClaim],
        missing_modalities: List[ModalityType] = None,
    ) -> ClinicalReport:
        """
        Synthesizes claims into a unified report without overwriting specialist evidence.
        """
        conflicts: List[ConflictLog] = []
        missing = missing_modalities or []

        # Simple cross-modal check example
        # If imaging indicates clear finding but notes contradict or remain uncertain
        for i in range(len(claims)):
            for j in range(i + 1, len(claims)):
                c1, c2 = claims[i], claims[j]
                if c1.modality != c2.modality:
                    # Detect potential conflict in uncertainty scores or findings
                    if abs(c1.uncertainty_score - c2.uncertainty_score) > self.conflict_threshold:
                        conflicts.append(
                            ConflictLog(
                                modality_a=c1.modality,
                                claim_a=c1.claim_text,
                                modality_b=c2.modality,
                                claim_b=c2.claim_text,
                                description=f"Uncertainty divergence between {c1.modality.value} ({c1.uncertainty_score:.2f}) and {c2.modality.value} ({c2.uncertainty_score:.2f})",
                            )
                        )

        # Compute aggregate confidence score (discounted if modalities are missing)
        avg_uncertainty = (
            sum(c.uncertainty_score for c in claims) / len(claims) if claims else 1.0
        )
        base_confidence = 1.0 - avg_uncertainty
        if missing:
            # Discount confidence when required modalities are absent
            base_confidence *= 0.8 ** len(missing)

        primary = (
            claims[0].claim_text if claims else "No conclusive findings available."
        )
        differentials = [c.claim_text for c in claims[1:]]

        return ClinicalReport(
            primary_impression=primary,
            differential_diagnoses=differentials,
            claims=claims,
            conflicts=conflicts,
            missing_modalities=missing,
            overall_confidence=round(base_confidence, 3),
        )
