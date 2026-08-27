"""
Clinical AI Agent: Unified Clinical LLM Client

Academic Citations & Foundations:
- Clinical LLM Multi-Turn Active Diagnostic Inquiry:
  * SDBench & MAI-DxO: "A Benchmark for Sequential Diagnosis with Language Models" (Microsoft Research, 2025; arXiv:2506.22405).
  * MedAgent-Pro: "Towards Evidence-Based Multi-Modal Medical Diagnosis via Reasoning Agentic Workflow" (Wang et al., ICLR 2026; arXiv:2503.18968).
  * Med-PaLM 2 / Med-Gemini: "Capabilities of Gemini Models in Medicine" (Google Research & DeepMind, 2024; arXiv:2404.18416).
"""

import os
import json
import time
import re
import logging
from typing import List, Dict, Any, Optional, Tuple, Callable

# Prevent LiteLLM from making remote network calls for cost maps
os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"

import litellm

from .schemas import (
    PatientCase,
    DifferentialHypothesis,
    SuspicionTier,
    ToolResult,
    FindingItem,
    ClinicalReport,
    DispositionType,
)
from .config import get_settings
from .prompts import (
    DIFFERENTIAL_PROMPT_TEMPLATE,
    ACTION_SELECTION_PROMPT_TEMPLATE,
    REPORT_SYNTHESIS_PROMPT_TEMPLATE,
)

# Suppress verbose LiteLLM logging
litellm.suppress_debug_info = True
litellm.set_verbose = False
litellm.telemetry = False
logging.getLogger("LiteLLM").setLevel(logging.ERROR)


def clean_json_string(text: str) -> str:
    """Cleans common LLM syntax irregularities like trailing commas or unescaped characters."""
    # Remove trailing commas before closing braces or brackets
    cleaned = re.sub(r",\s*([\]\}])", r"\1", text)
    # Remove single-line JS/JSON comments
    cleaned = re.sub(r"//.*?\n", "\n", cleaned)
    return cleaned


def extract_json_payload(raw_text: str) -> Dict[str, Any]:
    """Extracts JSON dictionary from raw model text or markdown fences with robust cleaning."""
    text = raw_text.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if match:
        text = match.group(1).strip()
    try:
        return json.loads(text)
    except Exception:
        pass

    # Extract outermost JSON object
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        snippet = text[start:end + 1]
        try:
            return json.loads(snippet)
        except Exception:
            pass

        try:
            cleaned = clean_json_string(snippet)
            return json.loads(cleaned)
        except Exception:
            pass

    # Fallback: attempt to parse with regex key-value extraction for structured reports
    extracted = {}
    if "primary_diagnosis" in text:
        m = re.search(r'"primary_diagnosis"\s*:\s*"([^"]+)"', text)
        if m:
            extracted["primary_diagnosis"] = m.group(1)
    if "disposition" in text:
        m = re.search(r'"disposition"\s*:\s*"([^"]+)"', text)
        if m:
            extracted["disposition"] = m.group(1)
    if "clinical_impression" in text:
        m = re.search(r'"clinical_impression"\s*:\s*"([^"]+)"', text)
        if m:
            extracted["clinical_impression"] = m.group(1)
    if "selected_action" in text:
        m = re.search(r'"selected_action"\s*:\s*"([^"]+)"', text)
        if m:
            extracted["selected_action"] = m.group(1)
    if "clinical_indication" in text:
        m = re.search(r'"clinical_indication"\s*:\s*"([^"]+)"', text)
        if m:
            extracted["clinical_indication"] = m.group(1)

    if extracted:
        return extracted

    raise ValueError(f"Could not parse valid JSON from text: {text[:150]}")



class ClinicalLLMClient:
    """
    Unified LLM client providing open-world clinical reasoning with token streaming:
    1. Pre-test differential formulation
    2. Active sequential diagnostic action / test selection
    3. Final evidence-grounded consultation report synthesis
    """

    def __init__(self, model_name: Optional[str] = None):
        settings = get_settings()
        self.model_name = model_name or settings.llm_model
        self.temperature = settings.llm_temperature
        self.max_tokens = settings.llm_max_tokens
        self.timeout = settings.llm_timeout

    def _call_completion(
        self,
        messages: List[Dict[str, str]],
        streaming_callback: Optional[Callable[[str], None]] = None,
        retries: int = 1,
    ) -> str:
        """Executes LLM chat completion with token streaming and retry handling."""
        last_error = None
        for attempt in range(retries + 1):
            try:
                if streaming_callback is not None:
                    response = litellm.completion(
                        model=self.model_name,
                        messages=messages,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        timeout=self.timeout,
                        stream=True,
                    )
                    full_text = []
                    for chunk in response:
                        delta = chunk.choices[0].delta.content or ""
                        if delta:
                            full_text.append(delta)
                            streaming_callback(delta)
                    return "".join(full_text)
                else:
                    response = litellm.completion(
                        model=self.model_name,
                        messages=messages,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        timeout=self.timeout,
                    )
                    return response.choices[0].message.content or ""
            except Exception as e:
                last_error = e
                if attempt < retries:
                    time.sleep(0.5)

        raise RuntimeError(f"LLM API completion failed on model '{self.model_name}': {last_error}")

    def formulate_initial_differential(
        self,
        case: PatientCase,
        streaming_callback: Optional[Callable[[str], None]] = None,
    ) -> List[DifferentialHypothesis]:
        """
        Formulates an open-world differential diagnosis from patient demographics,
        chief complaint, baseline vitals, and presenting history.
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
            raise RuntimeError(f"LLM did not return a valid 'differential' array in response: {raw_response[:200]}")

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

        return hypotheses

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
        tools_str = ", ".join(available_tools)

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

        raise RuntimeError(
            f"LLM selected invalid action '{action}'. Available tools: {available_tools}. Full response: {raw_response[:200]}"
        )

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
        top_condition = active_ddx[0].condition if active_ddx else "Undifferentiated Presentation"
        ddx_str = "\n".join([f"- {h.condition} [{h.tier.value}]: {h.clinical_rationale}" for h in active_ddx])
        evidence_str = "\n".join([f"- [{e.tool_name}]: {e.summary}" for e in evidence_history])

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
        
        raw_disp = str(data.get("disposition", "ADMIT_INPATIENT")).upper()
        if "DISCHARGE" in raw_disp:
            disposition = DispositionType.DISCHARGE_WITH_FOLLOWUP
        elif "ESCALATE" in raw_disp:
            disposition = DispositionType.ESCALATE_TO_PHYSICIAN
        else:
            disposition = DispositionType.ADMIT_INPATIENT

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

