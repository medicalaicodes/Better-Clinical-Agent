"""
Open-World Multimodal Clinical Diagnostic AI Agent
Interactive Demonstration and Benchmark Execution Script

Features:
- Complete real-time token streaming from the Attending LLM.
- 100% transparent turn-by-turn trace of:
  * Prompts and context sent to the LLM.
  * Intermediate reasoning and tool action selection.
  * Exact tool invocation arguments and raw neural model / laboratory outputs.
  * RAG query formulations, cosine similarity scores, and retrieved guideline monographs.
  * Final evidence-grounded consultation report and hospital disposition.

Usage:
  python demo_execution.py --case 0           # Run Case 0 (Streptococcal Pneumonia)
  python demo_execution.py --case 6           # Run Case 6 (Legionnaires' Disease)
  python demo_execution.py --case 0 --graph   # Run Case 0 via compiled LangGraph state machine
"""

import sys
import json
import argparse
from typing import Optional, List, Dict, Any

from src.clinical_agent.schemas import PatientCase, ClinicalReport, ToolResult
from src.clinical_agent.agent import AttendingClinicalAgent
from src.clinical_agent.graph import ClinicalAgentGraph
from src.clinical_agent.models import ensure_models_downloaded
from src.clinical_agent.evaluation.sample_cases import BENCHMARK_CASES
from src.clinical_agent.config import get_settings


def print_banner(title: str, subtitle: str = ""):
    print("\n" + "=" * 80)
    print(f"  {title.center(76)}")
    if subtitle:
        print(f"  {subtitle.center(76)}")
    print("=" * 80)


def print_case_intake(case_idx: int, case: PatientCase):
    print_banner(f"CLINICAL CASE {case_idx}: [{case.patient_id}]", "PATIENT EMERGENCY TRIAGE INTAKE")
    print(f"  Patient Demographics:  {case.age}-year-old {case.gender}")
    print(f"  Chief Complaint:       {case.chief_complaint}")
    print(f"  Triage Vital Signs:    {json.dumps(case.vitals)}")
    if case.clinical_notes:
        print(f"  Triage Doctor Notes:   {case.clinical_notes}")
    if case.image_path:
        print(f"  Attached Radiograph:   {case.image_path}")
    if case.available_labs:
        print(f"  Available Lab Panels:  {list(case.available_labs.keys())}")
    print("-" * 80 + "\n")


def stream_printer(token: str):
    """Streams LLM reasoning tokens in real-time to stdout."""
    sys.stdout.write(token)
    sys.stdout.flush()


def run_transparent_diagnostic_demo(case_idx: int, model_name: Optional[str] = None):
    if case_idx < 0 or case_idx >= len(BENCHMARK_CASES):
        print(f"Error: Invalid case index {case_idx}. Choose from 0 to {len(BENCHMARK_CASES) - 1}.")
        sys.exit(1)

    case = BENCHMARK_CASES[case_idx]
    print_case_intake(case_idx, case)

    agent = AttendingClinicalAgent(model_name=model_name)
    evidence_history: List[ToolResult] = []
    completed_tools: List[str] = []

    # =========================================================================
    # STAGE 1: Clinical Intake Assessment & Initial Differential Formulation
    # =========================================================================
    print_banner("STAGE 1: CLINICAL INTAKE & PRE-TEST DIFFERENTIAL REASONING", "Attending LLM: Natural Language Clinical Synthesis")
    print(f"-> Ingesting Patient Presentation & Notes: \"{case.clinical_notes}\"\n")
    print("-> Prompting Attending LLM with baseline vitals, demographics, and clinical history...")
    print("-> Streaming Model Reasoning:")
    print("   " + "-" * 74)

    active_ddx, intake_evidence = agent.llm_client.formulate_initial_differential(
        case=case,
        streaming_callback=stream_printer,
    )
    print("\n   " + "-" * 74)

    if intake_evidence:
        evidence_history.append(intake_evidence)
        print("\n-> LLM Extracted Pertinent Clinical Features & Negatives:")
        for f in intake_evidence.findings:
            status_tag = "POSITIVE" if f.status.value == "present" else "PERTINENT NEGATIVE"
            print(f"   * [{status_tag}] {f.name} (Confidence: {f.confidence:.2f})")
        print(f"-> Evidence Summary Appended to Ledger: {intake_evidence.summary}\n")

    print("-> Structured Pre-Test Differential Diagnoses:")
    for idx, ddx in enumerate(active_ddx, 1):
        tier_val = ddx.tier.value if hasattr(ddx.tier, "value") else str(ddx.tier)
        icd_str = f" ({ddx.icd10_code})" if ddx.icd10_code else ""
        print(f"   {idx}. [{tier_val}] {ddx.condition}{icd_str}")
        print(f"      Rationale: {ddx.clinical_rationale}")

    # =========================================================================
    # STAGE 2: Active Sequential Inquiry & Tool Orchestration Loop
    # =========================================================================

    max_turns = 4
    for turn in range(1, max_turns + 1):
        available_tools = [name for name in agent.tools.keys() if name not in completed_tools]
        if not available_tools:
            print_banner(f"TURN {turn}: WORKUP COMPLETE", "All specialist tools have been executed.")
            break

        print_banner(f"TURN {turn} OF {max_turns}: ACTIVE DIAGNOSTIC ACTION SELECTION", "Attending LLM Reasoning on Intermediate Evidence")
        print(f"-> Available Diagnostic Tools to Order: {available_tools}")
        print(f"-> Evidence Ledger Contains {len(evidence_history)} Recorded Findings")
        print("-> Streaming LLM Decision:")
        print("   " + "-" * 74)

        action, indication, search_query = agent.llm_client.select_next_action(
            case=case,
            active_ddx=active_ddx,
            evidence_history=evidence_history,
            available_tools=available_tools,
            streaming_callback=stream_printer,
        )
        print("\n   " + "-" * 74)

        print(f"\n-> Action Decided:      [{action}]")
        print(f"-> Medical Indication:  \"{indication}\"")
        if search_query:
            print(f"-> Formulated RAG Query: \"{search_query}\"")

        if action.upper() == "STOP":
            print("\n-> Attending Physician decided diagnostic evidence is sufficient to conclude workup.")
            break

        target_tool = agent.tools.get(action)
        if not target_tool:
            action = available_tools[0]
            target_tool = agent.tools[action]

        # Execute Tool & Display Transparent Outputs
        print(f"\n-> Invoking Specialist Tool: [{target_tool.name}] ...")
        tool_kwargs = {}
        if search_query:
            tool_kwargs["query"] = search_query

        tool_result = target_tool.run(case, **tool_kwargs)
        evidence_history.append(tool_result)
        completed_tools.append(action)

        print("\n-> Raw Specialist Tool Findings Emitted:")
        for f in tool_result.findings:
            val_str = f" = {f.quantitative_value} {f.unit}" if f.quantitative_value is not None else ""
            dev_val = f.details.get("deviation_percent")
            margin_val = f.details.get("margin")
            extra_str = ""
            if dev_val is not None:
                extra_str = f" [Dev: {dev_val:+.1f}%]"
            elif margin_val is not None:
                extra_str = f" [Margin: {margin_val:.3f}]"
            print(f"   * [{f.status.value}] {f.name}{val_str}{extra_str} (Conf: {f.confidence:.3f})")


        print(f"\n-> Appended to Evidence Ledger: \"{tool_result.summary}\"")

    # =========================================================================
    # STAGE 4: Grounded Consultation Report Synthesis
    # =========================================================================
    print_banner("STAGE 4: CLINICAL CONSULTATION REPORT SYNTHESIS", "Attending LLM Evidence Grounding & Triage Disposition")
    print("-> Prompting Attending LLM with full evidence ledger and guidelines context...")
    print("-> Streaming Consultation Synthesis:")
    print("   " + "-" * 74)

    report = agent.llm_client.synthesize_clinical_report(
        case=case,
        active_ddx=active_ddx,
        evidence_history=evidence_history,
        streaming_callback=stream_printer,
    )
    print("\n   " + "-" * 74)

    # Display Standard Final Report
    print_banner("FINAL DEFINITIVE CLINICAL REPORT & DISPOSITION", f"Patient ID: {report.patient_id}")
    print(f"  Primary Lead Diagnosis: {report.primary_diagnosis}")
    print(f"  Recommended Disposition: {report.disposition.value}\n")

    print("  Refined Differential Diagnosis:")
    for idx, ddx in enumerate(report.differential_diagnosis, 1):
        tier_str = ddx.tier.value if hasattr(ddx.tier, "value") else str(ddx.tier)
        print(f"    {idx}. [{tier_str}] {ddx.condition}")
        print(f"       Rationale: {ddx.clinical_rationale}")

    print("\n  Accumulated Evidence Trail:")
    for idx, ev in enumerate(report.evidence_trail, 1):
        print(f"    [{idx}] ({ev.tool_name}): {ev.summary}")

    print(f"\n  Clinical Impression:\n    {report.clinical_impression}")

    if report.recommendations:
        print("\n  Actionable Management Recommendations:")
        for rec in report.recommendations:
            print(f"    - {rec}")
    print("=" * 80 + "\n")


def run_graph_mode(case_idx: int, model_name: Optional[str] = None):
    case = BENCHMARK_CASES[case_idx]
    print_case_intake(case_idx, case)
    print_banner("LANGGRAPH STATE MACHINE EXECUTION", "Executing via Compiled State Graph")
    graph_runner = ClinicalAgentGraph(model_name=model_name)
    report = graph_runner.run(case)
    
    print_banner("FINAL CLINICAL CONSULTATION REPORT (LANGGRAPH)", f"Patient ID: {report.patient_id}")
    print(f"Primary Diagnosis:  {report.primary_diagnosis}")
    print(f"Disposition:        {report.disposition.value}")
    print(f"\nClinical Impression:\n{report.clinical_impression}")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Clinical AI Agent Demo and Benchmark Runner")
    parser.add_argument(
        "--case",
        type=int,
        default=None,
        help="Case index to execute (0 to 10). Example: --case 8 (Aortic Dissection stress test)",
    )
    parser.add_argument(
        "--graph",
        action="store_true",
        help="Run via LangGraph state machine instead of sequential agent loop",
    )
    parser.add_argument(
        "--stress-only",
        action="store_true",
        help="Run automated benchmark on only the 3 stress-test cases (Cases 8 to 10)",
    )
    parser.add_argument(
        "--canonical-only",
        action="store_true",
        help="Run automated benchmark on only the 8 canonical cases (Cases 0 to 7)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override the served LLM model name (default: LLM_MODEL from .env)",
    )
    parser.add_argument(
        "--download-models",
        action="store_true",
        help="Pre-download and verify all neural vision model checkpoints without running a case",
    )

    args = parser.parse_args()

    # Pre-download / verify foundation model checkpoints
    if args.download_models:
        ensure_models_downloaded(verbose=True)
        sys.exit(0)

    # Automatically ensure models are downloaded before executing cases
    ensure_models_downloaded(verbose=False)

    if args.stress_only:
        from src.clinical_agent.evaluation.sample_cases import STRESS_TEST_CASES, BENCHMARK_CASES
        stress_indices = [idx for idx, c in enumerate(BENCHMARK_CASES) if c in STRESS_TEST_CASES]
        print_banner("RUNNING FULL TRANSPARENT PIPELINE ON ALL STRESS-TEST CASES", f"Total Cases: {len(stress_indices)}")
        for idx in stress_indices:
            run_transparent_diagnostic_demo(case_idx=idx, model_name=args.model)
            print("\n" + "=" * 80 + "\n")
    elif args.canonical_only:
        from src.clinical_agent.evaluation.sample_cases import CANONICAL_CASES, BENCHMARK_CASES
        canonical_indices = [idx for idx, c in enumerate(BENCHMARK_CASES) if c in CANONICAL_CASES]
        print_banner("RUNNING FULL TRANSPARENT PIPELINE ON ALL CANONICAL CASES", f"Total Cases: {len(canonical_indices)}")
        for idx in canonical_indices:
            run_transparent_diagnostic_demo(case_idx=idx, model_name=args.model)
            print("\n" + "=" * 80 + "\n")
    elif args.graph:
        case_idx = args.case if args.case is not None else 0
        run_graph_mode(case_idx=case_idx, model_name=args.model)
    elif args.case is not None:
        run_transparent_diagnostic_demo(case_idx=args.case, model_name=args.model)
    else:
        print("No specific case provided. Running Case 0 (Streptococcal Pneumonia)...")
        print("Use --help to see all options (e.g. --case 8, --stress-only, --graph).\n")
        run_transparent_diagnostic_demo(case_idx=0, model_name=args.model)


if __name__ == "__main__":
    main()

