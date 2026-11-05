"""
Clinical AI Agent: Unified Clinical LLM Client

Academic Citations & Foundations:
- Clinical LLM Multi-Turn Active Diagnostic Inquiry:
  * SDBench & MAI-DxO: "A Benchmark for Sequential Diagnosis with Language Models" (Microsoft Research, 2025; arXiv:2506.22405).
  * MedAgent-Pro: "Towards Evidence-Based Multi-Modal Medical Diagnosis via Reasoning Agentic Workflow" (Wang et al., ICLR 2026; arXiv:2503.18968).
  * Med-PaLM 2 / Med-Gemini: "Capabilities of Gemini Models in Medicine" (Google Research & DeepMind, 2024; arXiv:2404.18416).
"""

import json
from typing import Any, Callable, Dict, List, Optional, Tuple

from .llm import LLM, parse_json
from .prompts import (
    ACTION_SELECTION_PROMPT_TEMPLATE,
    DIFFERENTIAL_PROMPT_TEMPLATE,
    REPORT_SYNTHESIS_PROMPT_TEMPLATE,
)
from .schemas import (
    ClinicalReport,
    DifferentialHypothesis,
    DispositionType,
    FindingItem,
    FindingStatus,
    ModalityType,
    PatientCase,
    SuspicionTier,
    ToolResult,
)


def extract_json_payload(raw_text: str) -> Dict[str, Any]:
    data = parse_json(raw_text)
    if data is None:
        raise ValueError(f"Could not parse JSON from LLM output: {raw_text[:150]}")
    return data


class ClinicalLLMClient:
    """
    Unified LLM client providing open-world clinical reasoning with token streaming:
    1. Pre-test differential formulation
    2. Active sequential diagnostic action / test selection
    3. Final evidence-grounded consultation report synthesis
    """

    def __init__(self, model_name: Optional[str] = None):
        self.llm = LLM(model=model_name)
        self.model_name = self.llm.model

    def _call_completion(self, messages: List[Dict[str, str]],
                         streaming_callback: Optional[Callable[[str], None]] = None) -> str:
        return self.llm.chat(messages[-1]["content"], on_token=streaming_callback)

    def formulate_initial_differential(
        self,
        case: PatientCase,
        streaming_callback: Optional[Callable[[str], None]] = None,
    ) -> Tuple[List[DifferentialHypothesis], Optional[ToolResult]]:
        """
        Formulates an open-world differential diagnosis and extracts structured
        pertinent positives and pertinent negatives directly from clinical notes.
        """
        prompt = DIFFERENTIAL_PROMPT_TEMPLATE.format(
            age=case.age,
            gender=case.gender,
            chief_complaint=case.chief_complaint,
            vitals=json.dumps(case.vitals),
            clinical_notes=case.clinical_notes or "None",
        )
        raw_response = self._call_completion(
            [{"role": "user", "content": prompt}],
            streaming_callback=streaming_callback,
        )
        data = extract_json_payload(raw_response)
        items = data.get("differential", [])
        if not items or not isinstance(items, list):
            # Attempt to extract if differential is nested or root array
            if isinstance(data, list):
                items = data
            else:
                items = [{"condition": "Undifferentiated Presentation", "tier": "MODERATE_SUSPICION", "clinical_rationale": "Intake evaluation."}]

        hypotheses = []
        for item in items:
            raw_tier = str(item.get("tier", "MODERATE_SUSPICION")).upper()
            if "HIGH" in raw_tier:
                tier = SuspicionTier.HIGH_SUSPICION
            elif "RULE" in raw_tier or "LOW" in raw_tier:
                tier = SuspicionTier.RULE_OUT
            else:
                tier = SuspicionTier.MODERATE_SUSPICION

            hypotheses.append(
                DifferentialHypothesis(
                    condition=item.get("condition", "Undifferentiated Presentation"),
                    icd10_code=item.get("icd10_code"),
                    tier=tier,
                    clinical_rationale=item.get("clinical_rationale", "Formulated from patient presentation."),
                )
            )

        # Build structured intake findings from LLM's natural language comprehension
        findings: List[FindingItem] = []
        raw_positives = data.get("pertinent_positives", [])
        raw_negatives = data.get("pertinent_negatives", [])

        # Programmatic case-insensitive deduplication (preserves order)
        def _dedupe(items: Any) -> List[str]:
            if not isinstance(items, list):
                return []
            seen = set()
            out = []
            for x in items:
                if isinstance(x, str):
                    clean = x.strip()
                    key = clean.lower()
                    if key and key not in seen:
                        seen.add(key)
                        out.append(clean)
            return out

        positives = _dedupe(raw_positives)
        negatives = _dedupe(raw_negatives)

        for pos in positives:
            findings.append(
                FindingItem(
                    name=f"Clinical Sign: {pos}",
                    status=FindingStatus.PRESENT,
                    confidence=0.98,
                    details={"category": "pertinent_positive"},
                )
            )
        for neg in negatives:
            findings.append(
                FindingItem(
                    name=f"Pertinent Negative: {neg}",
                    status=FindingStatus.ABSENT,
                    confidence=0.98,
                    details={"category": "pertinent_negative"},
                )
            )


        intake_result = None
        if findings or case.clinical_notes:
            pos_count = len([f for f in findings if f.status == FindingStatus.PRESENT])
            neg_count = len([f for f in findings if f.status == FindingStatus.ABSENT])
            summary_parts = []
            if pos_count:
                summary_parts.append(f"Identified {pos_count} positive clinical feature(s)")
            if neg_count:
                summary_parts.append(f"Confirmed {neg_count} pertinent negative(s)")
            summary_str = f"Clinical Intake Assessment: {', '.join(summary_parts) if summary_parts else 'History reviewed'}."

            intake_result = ToolResult(
                tool_name="clinical_history_synthesis",
                modality=ModalityType.CLINICAL_NOTES,
                findings=findings,
                summary=summary_str,
                raw_output={"pertinent_positives": positives, "pertinent_negatives": negatives, "raw_notes": case.clinical_notes},
            )

        return hypotheses, intake_result


    def select_next_action(
        self,
        case: PatientCase,
        active_ddx: List[DifferentialHypothesis],
        evidence_history: List[ToolResult],
        available_tools: List[str],
        streaming_callback: Optional[Callable[[str], None]] = None,
    ) -> Tuple[str, str, Optional[str]]:
        """
        Evaluates the current diagnostic state and determines the next highest-yield
        diagnostic test to order, returning (action, indication, search_query).
        """
        if not available_tools:
            return "STOP", "All diagnostic tools have been exhausted.", None

        ddx_str = "\n".join([f"- {h.condition} [{h.tier.value}]: {h.clinical_rationale}" for h in active_ddx])
        evidence_str = "\n".join([f"- [Turn {i+1} - {e.tool_name} ({e.modality.value})]: {e.summary}" for i, e in enumerate(evidence_history)]) or "No tests ordered yet."

        TOOL_CAPABILITY_MAP = {
            "chest_xray_classifier": "Deep learning radiograph classifier evaluating 18 thoracic pathologies and parenchymal opacities on chest images.",
            "medsam_segmenter": "Vision Transformer anatomical segmenter calculating lung volume symmetry, collapse percentage, and Cardiothoracic Ratio (CTR) on chest images.",
            "laboratory_tests": "Quantitative blood analyzer measuring inflammatory, cardiac, thrombotic, hematologic, and metabolic analyte deviations.",
            "medical_knowledge_rag": "Semantic retrieval engine querying indexed clinical practice guidelines and disease monographs.",
        }
        tools_str = "\n".join([f"- '{t}': {TOOL_CAPABILITY_MAP.get(t, 'Specialist diagnostic tool.')}" for t in available_tools])

        prompt = ACTION_SELECTION_PROMPT_TEMPLATE.format(
            age=case.age,
            gender=case.gender,
            chief_complaint=case.chief_complaint,
            vitals=json.dumps(case.vitals),
            differential_str=ddx_str,
            evidence_str=evidence_str,
            tools_str=tools_str,
        )


        raw_response = self._call_completion(
            [{"role": "user", "content": prompt}],
            streaming_callback=streaming_callback,
        )
        data = extract_json_payload(raw_response)
        action = data.get("selected_action", "").strip()
        indication = data.get("clinical_indication", "Ordered to evaluate diagnostic differential.").strip()
        search_query = data.get("search_query", "").strip() or None

        if action.upper() == "STOP":
            return "STOP", indication, None

        for tool in available_tools:
            if tool.lower() == action.lower() or action.lower() in tool.lower():
                return tool, indication, search_query

        # Graceful fallback: If LLM re-requested an already-run tool or hallucinatory name, select remaining available tool or STOP
        if available_tools:
            return available_tools[0], indication, search_query
        return "STOP", indication, None


    def synthesize_clinical_report(
        self,
        case: PatientCase,
        active_ddx: List[DifferentialHypothesis],
        evidence_history: List[ToolResult],
        streaming_callback: Optional[Callable[[str], None]] = None,
    ) -> ClinicalReport:
        """
        Synthesizes the comprehensive final clinical consultation report,
        refined differential, clinical impression prose, recommendations, and disposition.
        """
        ddx_str = "\n".join([f"- {h.condition} [{h.tier.value}]: {h.clinical_rationale}" for h in active_ddx])
        
        completed_tests = [e.tool_name for e in evidence_history]
        completed_note = f"[Completed Diagnostic Evaluations on Record: {', '.join(completed_tests)}]\n"
        evidence_str = completed_note + "\n".join([f"- [{e.tool_name}]: {e.summary}" for e in evidence_history])

        prompt = REPORT_SYNTHESIS_PROMPT_TEMPLATE.format(
            age=case.age,
            gender=case.gender,
            chief_complaint=case.chief_complaint,
            vitals=json.dumps(case.vitals),
            differential_str=ddx_str,
            evidence_str=evidence_str,
        )


        raw_response = self._call_completion(
            [{"role": "user", "content": prompt}],
            streaming_callback=streaming_callback,
        )
        data = extract_json_payload(raw_response)
        primary_diag = data.get("primary_diagnosis")
        if not primary_diag:
            raise RuntimeError(f"LLM synthesis missing 'primary_diagnosis'. Raw response: {raw_response[:200]}")
        
        raw_disp = str(data.get("disposition", "ESCALATE_TO_PHYSICIAN")).upper()
        if "DISCHARGE" in raw_disp:
            disposition = DispositionType.DISCHARGE_WITH_FOLLOWUP
        elif "ADMIT" in raw_disp:
            disposition = DispositionType.ADMIT_INPATIENT
        else:  # missing or unrecognised -> hand to a human, never silently admit/discharge
            disposition = DispositionType.ESCALATE_TO_PHYSICIAN

        impression = data.get("clinical_impression", "")
        if not impression:
            raise RuntimeError(f"LLM synthesis missing 'clinical_impression'. Raw response: {raw_response[:200]}")
        
        recommendations = data.get("recommendations", [])
        
        # Parse refined differential
        parsed_ddx = data.get("differential_diagnosis", [])
        final_ddx = []
        if parsed_ddx and isinstance(parsed_ddx, list):
            for item in parsed_ddx:
                if isinstance(item, dict):
                    raw_tier = str(item.get("tier", "MODERATE_SUSPICION")).upper()
                    if "HIGH" in raw_tier:
                        tier = SuspicionTier.HIGH_SUSPICION
                    elif "RULE" in raw_tier or "LOW" in raw_tier:
                        tier = SuspicionTier.RULE_OUT
                    else:
                        tier = SuspicionTier.MODERATE_SUSPICION
                    final_ddx.append(
                        DifferentialHypothesis(
                            condition=item.get("condition", "Clinical Candidate"),
                            icd10_code=item.get("icd10_code"),
                            tier=tier,
                            clinical_rationale=item.get("clinical_rationale", "Evidence-grounded hypothesis."),
                        )
                    )
        if not final_ddx:
            final_ddx = list(active_ddx)

        all_findings: List[FindingItem] = []
        for res in evidence_history:
            all_findings.extend(res.findings)

        return ClinicalReport(
            case_id=case.patient_id,
            patient_id=case.patient_id,
            primary_diagnosis=primary_diag,
            disposition=disposition,
            differential_diagnosis=final_ddx,
            key_findings=all_findings,
            clinical_impression=impression,
            recommendations=recommendations,
            evidence_trail=evidence_history,
        )

