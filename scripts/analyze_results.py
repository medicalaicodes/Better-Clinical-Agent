"""
Looks inside the saved results to see WHY methods win or lose.

    python scripts/analyze_results.py --dataset medqa
    python scripts/analyze_results.py --dataset medqa_dev --run-name demo2

Sections (all on the questions that every compared method answered):
  1. Accuracy of every method, split by how strongly the 5 self-consistency votes agree.
     This is where "retrieval hurts when the model is sure, helps when it isn't" comes from.
  2. Accuracy per subject (MMLU-Med question ids carry the subject).
  3. Which questions each method fixed or broke compared with the baseline.
  4. For the agent: did the loop fix or break its own first answer, what did the critic do.
"""

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.clinical_agent.config import get_settings  # noqa: E402
from src.clinical_agent.evaluation.mirage import read_results  # noqa: E402


def percent(correct, total):
    if total == 0:
        return "-"
    return f"{100 * correct / total:.1f} ({correct}/{total})"


def load_run(run_dir):
    """{method name: {question id: result row}} for every result file in the folder."""
    runs = {}
    for file_name in sorted(os.listdir(run_dir)):
        if not file_name.endswith(".jsonl"):
            continue
        rows = read_results(os.path.join(run_dir, file_name))
        if rows and "correct" in list(rows.values())[0]:      # skip files that aren't method results
            runs[file_name[:-len(".jsonl")]] = rows
    return runs


def top_vote_count(votes):
    """How many of the 5 votes went to the most popular letter (5 = all agree)."""
    best = 0
    for letter, count in votes.items():
        if letter in ("None", "null"):
            continue                       # an answer we couldn't read isn't a vote for anything
        if count > best:
            best = count
    return best


def section_agreement(runs, ids):
    print("## 1. Accuracy by how much the 5 self-consistency votes agree\n")
    if "cot_sc" not in runs:
        print("(needs cot_sc results in this folder)\n")
        return

    groups = {}                            # 5 -> [question ids], 4 -> [...], ...
    for question_id in ids:
        agreement = top_vote_count(runs["cot_sc"][question_id]["trace"].get("votes", {}))
        if agreement not in groups:
            groups[agreement] = []
        groups[agreement].append(question_id)

    names = list(runs)
    print("| votes agreeing | questions | " + " | ".join(names) + " |")
    print("|---|---|" + "---|" * len(names))
    for agreement in sorted(groups, reverse=True):
        group = groups[agreement]
        cells = []
        for name in names:
            correct = 0
            for question_id in group:
                if runs[name][question_id]["correct"]:
                    correct += 1
            cells.append(f"{100 * correct / len(group):.1f}")
        print(f"| {agreement} of 5 | {len(group)} | " + " | ".join(cells) + " |")
    print()


def section_subjects(runs, ids):
    """MMLU-Med ids look like 'anatomy-000' or 'college_medicine-012'."""
    subjects = {}
    for question_id in ids:
        match = re.match(r"([a-z_]+)-\d+$", question_id)
        if match:
            subject = match.group(1)
            if subject not in subjects:
                subjects[subject] = []
            subjects[subject].append(question_id)
    if len(subjects) < 2:
        return

    print("## 2. Accuracy per subject\n")
    names = list(runs)
    print("| subject | n | " + " | ".join(names) + " |")
    print("|---|---|" + "---|" * len(names))
    for subject in sorted(subjects):
        group = subjects[subject]
        cells = []
        for name in names:
            correct = 0
            for question_id in group:
                if runs[name][question_id]["correct"]:
                    correct += 1
            cells.append(f"{100 * correct / len(group):.1f}")
        print(f"| {subject} | {len(group)} | " + " | ".join(cells) + " |")
    print()


def section_fixed_broke(runs, ids, baseline, out_path):
    print(f"## 3. Questions each method fixed / broke compared with `{baseline}`\n")
    lists = {}
    for name in runs:
        if name == baseline:
            continue
        fixed = []
        broke = []
        for question_id in ids:
            this_right = runs[name][question_id]["correct"]
            base_right = runs[baseline][question_id]["correct"]
            if this_right and not base_right:
                fixed.append(question_id)
            if base_right and not this_right:
                broke.append(question_id)
        lists[name] = {"fixed": fixed, "broke": broke}
        print(f"- **{name}**: fixed {len(fixed)}, broke {len(broke)}. First broken: {', '.join(broke[:8]) or '-'}")

    with open(out_path, "w") as f:
        json.dump(lists, f, indent=1)
    print(f"\nFull lists: `{out_path}`\n")


def section_agent(name, rows):
    """The agent compared with its OWN first answer, before it saw any evidence."""
    print(f"## 4. Agent loop: `{name}`\n")
    first_right = 0
    first_wrong = 0
    broke = 0              # first answer right, final answer wrong
    fixed = 0              # first answer wrong, final answer right
    critic_acted = 0
    with_open_issues = 0
    right_with_open_issues = 0
    fallbacks = 0
    turns = 0
    critiques = 0

    for row in rows.values():
        trace = row["trace"]
        turns += trace.get("turns", 0)
        critiques += trace.get("critiques", 0)
        if trace.get("mapping_fallback"):
            fallbacks += 1

        if trace.get("first", {}).get("choice") == row["gold"]:
            first_right += 1
            if not row["correct"]:
                broke += 1
        else:
            first_wrong += 1
            if row["correct"]:
                fixed += 1

        for event in trace.get("events", []):
            if event.get("event") == "critique" and (event.get("blocking") or event.get("new_hypotheses")):
                critic_acted += 1
                break

        if trace.get("open_issues"):
            with_open_issues += 1
            if row["correct"]:
                right_with_open_issues += 1

    n = max(1, len(rows))
    print(f"- First answer (no options, before any evidence) right: {percent(first_right, len(rows))}")
    print(f"- Loop broke a right first answer: **{broke}** of {first_right}")
    print(f"- Loop fixed a wrong first answer: **{fixed}** of {first_wrong}")
    print(f"- Critic raised a serious objection or a new diagnosis: {critic_acted} questions")
    print(f"- Ended with an open doubt: {with_open_issues}, accuracy on those {percent(right_with_open_issues, with_open_issues)}")
    print(f"- No option matched the free-text answer (fallback used): {fallbacks}")
    print(f"- Average turns {turns / n:.2f}, critiques {critiques / n:.2f}\n")


def main():
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="medqa")
    parser.add_argument("--run-name", default=settings.llm_model.replace("/", "_"))
    parser.add_argument("--baseline", default="cot")
    args = parser.parse_args()

    run_dir = os.path.join("results", args.run_name, args.dataset)
    runs = load_run(run_dir)
    if not runs:
        sys.exit(f"No results in {run_dir}")
    baseline = args.baseline
    if baseline not in runs:
        baseline = list(runs)[0]

    # Only questions that every method answered, so all numbers are comparable.
    common = None
    for rows in runs.values():
        if common is None:
            common = set(rows)
        else:
            common = common & set(rows)
    ids = sorted(common)

    print(f"# Error analysis: {args.dataset}, {args.run_name} ({len(ids)} questions answered by all methods)\n")
    section_agreement(runs, ids)
    section_subjects(runs, ids)
    section_fixed_broke(runs, ids, baseline, os.path.join(run_dir, "fixed_broke.json"))
    for name in runs:
        if name.startswith("dx_agent"):
            only_common = {}
            for question_id in ids:
                only_common[question_id] = runs[name][question_id]
            section_agent(name, only_common)


if __name__ == "__main__":
    main()
