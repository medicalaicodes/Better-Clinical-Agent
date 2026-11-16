"""
Live demo of the clinical agent on one of the sample patients, printing every step as it happens:
the model's reasoning word by word, each test it orders, what each test finds, and the final report.

    python demo_execution.py --case 0       # pneumonia
    python demo_execution.py --case 8       # aortic dissection stress test (should escalate)

Needs the LLM server running (see scripts/serve_vllm.sh). The image models download themselves
the first time they're used.
"""

import argparse
import json
import sys

from src.clinical_agent.agent import AttendingClinicalAgent
from src.clinical_agent.evaluation.sample_cases import BENCHMARK_CASES


def banner(title):
    print("\n" + "=" * 80)
    print("  " + title)
    print("=" * 80)


def print_token(token):
    # Called for every piece of text the model writes, so the reasoning appears live.
    sys.stdout.write(token)
    sys.stdout.flush()


def print_event(event, details):
    # Called by the agent after each step.
    if event == "intake":
        print("\n\nFindings from the notes:")
        for finding in details["intake"].findings:
            print(f"   * {finding.name}")
        print("First differential:")
        for hypothesis in details["differential"]:
            print(f"   - [{hypothesis.tier.value}] {hypothesis.condition}: {hypothesis.clinical_rationale}")

    elif event == "choice":
        if details["next_tool"] == "STOP":
            banner("ENOUGH EVIDENCE - WRITING THE REPORT")
        else:
            banner(f"TURN {details['turns']}: ORDERING {details['next_tool']}")
            print(f"Why: {details['reason']}")
            if details["query"]:
                print(f"Library search: {details['query']}")

    elif event == "tool_result":
        print("\nWhat the tool found:")
        for finding in details.findings:
            value = ""
            if finding.quantitative_value is not None:
                value = f" = {finding.quantitative_value} {finding.unit or ''}"
            print(f"   * [{finding.status.value}] {finding.name}{value}")
        print(f"Summary: {details.summary}")


def main():
    parser = argparse.ArgumentParser(description="Live demo of the clinical agent")
    parser.add_argument("--case", type=int, default=0, help=f"which patient, 0 to {len(BENCHMARK_CASES) - 1}")
    parser.add_argument("--model", default=None, help="served model name (default: LLM_MODEL from .env)")
    parser.add_argument("--max-turns", type=int, default=4)
    args = parser.parse_args()

    case = BENCHMARK_CASES[args.case]
    banner(f"PATIENT {args.case}: {case.patient_id}")
    print(f"{case.age}-year-old {case.gender}. Complaint: {case.chief_complaint}")
    print(f"Vitals: {json.dumps(case.vitals)}")
    print(f"Notes: {case.clinical_notes}")
    print(f"X-ray: {case.image_path}   Labs available: {list(case.available_labs)}")
    banner("INTAKE: THE MODEL READS THE CASE")

    agent = AttendingClinicalAgent(model_name=args.model)
    agent.on_token = print_token
    agent.on_event = print_event
    report = agent.diagnose(case, max_turns=args.max_turns)

    banner("FINAL REPORT")
    print(f"Diagnosis:   {report.primary_diagnosis}")
    print(f"Disposition: {report.disposition.value}")
    print("Differential:")
    for hypothesis in report.differential_diagnosis:
        print(f"   - [{hypothesis.tier.value}] {hypothesis.condition}: {hypothesis.clinical_rationale}")
    print(f"Impression:  {report.clinical_impression}")
    print("Recommendations:")
    for recommendation in report.recommendations:
        print(f"   - {recommendation}")


if __name__ == "__main__":
    main()
