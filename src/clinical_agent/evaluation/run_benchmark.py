"""
Clinical AI Agent: Benchmark Evaluation Runner

Academic Citations & Foundations:
- SDBench: "A Benchmark for Sequential Diagnosis with Language Models" (Microsoft Research, 2025; arXiv:2506.22405).
- MedAgent-Pro: "Towards Evidence-Based Multi-Modal Medical Diagnosis via Reasoning Agentic Workflow" (Wang et al., ICLR 2026; arXiv:2503.18968).
"""

from typing import List, Dict, Any, Optional
from ..agent import AttendingClinicalAgent
from ..schemas import PatientCase
from .sample_cases import BENCHMARK_CASES
from ..config import get_settings


def run_clinical_benchmark(model_name: Optional[str] = None, verbose: bool = True) -> Dict[str, Any]:
    """Executes sequential benchmark evaluation across all 8 canonical benchmark cases."""
    if model_name is None:
        model_name = get_settings().llm_model

    agent = AttendingClinicalAgent(model_name=model_name)
    results = []

    if verbose:
        print("\n" + "=" * 70)
        print("    OPEN-WORLD MULTIMODAL CLINICAL AGENT - SEQUENTIAL EVALUATION")
        print("=" * 70)
        print(f"Attending LLM Model: {model_name}\n")

    for idx, case in enumerate(BENCHMARK_CASES):
        if verbose:
            print(f"\n--- [EVALUATING CASE {idx}: {case.patient_id}] ---")
            print(f"  Demographics:    {case.age}yo {case.gender}")
            print(f"  Chief Complaint: {case.chief_complaint}")
            print(f"  Vitals:          {case.vitals}")

        report = agent.diagnose(case=case, max_turns=4)

        result_entry = {
            "case_idx": idx,
            "patient_id": case.patient_id,
            "chief_complaint": case.chief_complaint,
            "primary_diagnosis": report.primary_diagnosis,
            "disposition": report.disposition.value,
            "evidence_count": len(report.evidence_trail),
            "evidence_trail": [
                {
                    "tool": ev.tool_name,
                    "summary": ev.summary,
                    "findings_count": len(ev.findings),
                }
                for ev in report.evidence_trail
            ],
            "differential": [
                {"condition": d.condition, "tier": d.tier.value, "rationale": d.clinical_rationale}
                for d in report.differential_diagnosis
            ],
            "clinical_impression": report.clinical_impression,
            "recommendations": report.recommendations,
        }
        results.append(result_entry)

        if verbose:
            print(f"  -> Lead Diagnosis: {report.primary_diagnosis}")
            print(f"  -> Disposition:    {report.disposition.value}")
            print(f"  -> Evidence Trail: {len(report.evidence_trail)} packets:")
            for ev_idx, ev in enumerate(report.evidence_trail, 1):
                print(f"     [{ev_idx}] {ev.tool_name}: {ev.summary[:85]}...")
            print(f"  -> Clinical Impression:\n     {report.clinical_impression[:160]}...")
            print("-" * 70)

    return {"total_cases": len(BENCHMARK_CASES), "results": results}


if __name__ == "__main__":
    run_clinical_benchmark(verbose=True)

