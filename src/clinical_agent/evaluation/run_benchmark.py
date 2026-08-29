"""
Clinical AI Agent: Benchmark Evaluation Runner

Academic Citations & Foundations:
- SDBench: "A Benchmark for Sequential Diagnosis with Language Models" (Microsoft Research, 2025; arXiv:2506.22405).
- MedAgent-Pro: "Towards Evidence-Based Multi-Modal Medical Diagnosis via Reasoning Agentic Workflow" (Wang et al., ICLR 2026; arXiv:2503.18968).
"""

import argparse
from typing import List, Dict, Any, Optional
from ..agent import AttendingClinicalAgent
from ..schemas import PatientCase
from .sample_cases import BENCHMARK_CASES, CANONICAL_CASES, STRESS_TEST_CASES
from ..config import get_settings


def run_clinical_benchmark(
    model_name: Optional[str] = None,
    cases: Optional[List[PatientCase]] = None,
    suite_name: str = "FULL (Canonical + Stress-Test)",
    verbose: bool = True,
) -> Dict[str, Any]:
    """Executes sequential benchmark evaluation across canonical and stress-test clinical cases."""
    if model_name is None:
        model_name = get_settings().llm_model

    eval_cases = cases or BENCHMARK_CASES
    agent = AttendingClinicalAgent(model_name=model_name)
    results = []

    if verbose:
        print("\n" + "=" * 80)
        print(f"  OPEN-WORLD MULTIMODAL CLINICAL AGENT - SEQUENTIAL EVALUATION SUITE")
        print(f"  Evaluation Cohort: {suite_name} ({len(eval_cases)} cases)")
        print(f"  Attending Model:   {model_name}")
        print("=" * 80)

    for idx, case in enumerate(eval_cases):
        is_stress = idx >= len(CANONICAL_CASES) if cases is None else (case in STRESS_TEST_CASES)
        case_tag = "STRESS-TEST / UNCERTAINTY" if is_stress else "CANONICAL MULTIMODAL"

        if verbose:
            print(f"\n--- [EVALUATING CASE {idx}: {case.patient_id} ({case_tag})] ---")
            print(f"  Demographics:    {case.age}yo {case.gender}")
            print(f"  Chief Complaint: {case.chief_complaint}")
            print(f"  Vitals:          {case.vitals}")

        report = agent.diagnose(case=case, max_turns=4)

        result_entry = {
            "case_idx": idx,
            "patient_id": case.patient_id,
            "case_type": case_tag,
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
            print("-" * 80)

    if verbose:
        print("\n" + "=" * 80)
        print("  EVALUATION SUMMARY & DISPOSITION MATRIX")
        print("=" * 80)
        admit_count = sum(1 for r in results if r["disposition"] == "ADMIT_INPATIENT")
        discharge_count = sum(1 for r in results if r["disposition"] == "DISCHARGE_WITH_FOLLOWUP")
        escalate_count = sum(1 for r in results if r["disposition"] == "ESCALATE_TO_PHYSICIAN")
        print(f"  Total Cases Evaluated:       {len(results)}")
        print(f"  - ADMIT_INPATIENT:           {admit_count}")
        print(f"  - DISCHARGE_WITH_FOLLOWUP:   {discharge_count}")
        print(f"  - ESCALATE_TO_PHYSICIAN:     {escalate_count}")
        print("=" * 80 + "\n")

    return {"total_cases": len(eval_cases), "results": results}


def main():
    parser = argparse.ArgumentParser(description="Clinical Agent Benchmark Runner")
    parser.add_argument("--model", type=str, default=None, help="LLM Model name override")
    parser.add_argument("--canonical-only", action="store_true", help="Run only the 8 canonical cases")
    parser.add_argument("--stress-only", action="store_true", help="Run only the 3 stress-test cases")
    args = parser.parse_args()

    if args.canonical_only:
        run_clinical_benchmark(model_name=args.model, cases=CANONICAL_CASES, suite_name="Canonical Cohort", verbose=True)
    elif args.stress_only:
        run_clinical_benchmark(model_name=args.model, cases=STRESS_TEST_CASES, suite_name="Stress-Test Cohort", verbose=True)
    else:
        run_clinical_benchmark(model_name=args.model, verbose=True)


if __name__ == "__main__":
    main()
