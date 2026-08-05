"""
Tests for Clinical Agent Schemas and Evidence Coordinator
"""

import unittest
from src.clinical_agent import (
    ModalityType,
    EvidencePacket,
    AgentClaim,
    SpecialistRouter,
    EvidenceCoordinator,
)


class TestClinicalAgent(unittest.TestCase):

    def test_router_missing_modalities(self):
        router = SpecialistRouter(
            required_modalities=[ModalityType.CXR, ModalityType.EHR, ModalityType.NOTES]
        )
        # Pass only EHR data
        result = router.route_inputs({ModalityType.EHR: "patient_data"})
        self.assertIn(ModalityType.EHR, result["available_modalities"])
        self.assertIn(ModalityType.CXR, result["missing_modalities"])
        self.assertIn(ModalityType.NOTES, result["missing_modalities"])

    def test_evidence_coordination_and_conflict_detection(self):
        evidence1 = EvidencePacket(
            modality=ModalityType.CXR,
            findings=["Lobar consolidation in right lower lobe"],
            confidence=0.9,
        )
        claim1 = AgentClaim(
            agent_id="cxr_specialist",
            modality=ModalityType.CXR,
            claim_text="Right lower lobe pneumonia",
            supporting_evidence=[evidence1],
            uncertainty_score=0.1,  # Low uncertainty
        )

        evidence2 = EvidencePacket(
            modality=ModalityType.NOTES,
            findings=["Cough reported, non-feverish"],
            confidence=0.5,
        )
        claim2 = AgentClaim(
            agent_id="notes_specialist",
            modality=ModalityType.NOTES,
            claim_text="Mild viral bronchitis",
            supporting_evidence=[evidence2],
            uncertainty_score=0.7,  # High uncertainty
        )

        coordinator = EvidenceCoordinator(conflict_threshold=0.5)
        report = coordinator.synthesize(
            claims=[claim1, claim2],
            missing_modalities=[ModalityType.GENOMICS],
        )

        self.assertEqual(report.primary_impression, "Right lower lobe pneumonia")
        self.assertIn(ModalityType.GENOMICS, report.missing_modalities)
        self.assertGreater(len(report.conflicts), 0)
        self.assertLess(report.overall_confidence, 1.0)


if __name__ == "__main__":
    unittest.main()
