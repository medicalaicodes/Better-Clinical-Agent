"""
Error analysis for MIRAGE runs: looks inside the traces to see WHY methods win or lose.

    python scripts/analyze_results.py --dataset medqa                 # default run folder = LLM_MODEL
    python scripts/analyze_results.py --dataset mmlu --run-name llama31-8b > analysis_mmlu.md

Sections (all computed on questions answered by every method that is compared):
  1. Does the gate's signal (closed-book vote agreement) predict correctness?
  2. DDx-RAG light vs full path: accuracy, and how CoT did on the same questions.
  3. Adjudication: when DDx-RAG overruled the closed-book majority, was it right to?
  4. Which evidence intents ended up in the prompt, for right vs wrong answers.
  5. Per-subject accuracy (MMLU-Med has 6 subjects in the question id).
  6. Question ids that each method fixed / broke relative to CoT, for manual reading.
  7. DxAgent: harm / recovery against its own first answer, critic activity, open issues.
"""

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.clinical_agent.config import get_settings  # noqa: E402
from src.clinical_agent.evaluation.mirage import read_results  # noqa: E402


def pct(correct, total):
    if total == 0:
        return "-"
    return f"{100 * correct / total:.1f} ({correct}/{total})"


def load_run(run_dir):
    """method name (file name without .jsonl) -> {question id -> result row}"""
    runs = {}
    for file_name in sorted(os.listdir(run_dir)):
        if file_name.endswith(".jsonl"):
            rows = read_results(os.path.join(run_dir, file_name))
            if rows and "correct" in next(iter(rows.values())):     # skip non-result files
                runs[file_name[:-len(".jsonl")]] = rows
    return runs


def section_gate_signal(ddx, cot):
    """If agreement is a good confidence signal, accuracy should rise with agreement."""
    print("## 1. Is closed-book vote agreement a good confidence signal?\n")
    print("| agreement | questions | majority vote correct | CoT correct | DDx-RAG correct |")
    print("|---|---|---|---|---|")

    groups = defaultdict(list)
    for qid, row in ddx.items():
        groups[row["trace"].get("agreement", 0.0)].append(qid)

    for agreement in sorted(groups, reverse=True):
        ids = groups[agreement]
        majority_correct = 0
        for qid in ids:
            votes = ddx[qid]["trace"].get("votes", {})
            letters = {k: v for k, v in votes.items() if k not in ("None", "null")}
            if letters and max(letters, key=lambda k: (letters[k], k)) == ddx[qid]["gold"]:
                majority_correct += 1
        cot_correct = sum(1 for qid in ids if qid in cot and cot[qid]["correct"])
        ddx_correct = sum(1 for qid in ids if ddx[qid]["correct"])
        print(f"| {agreement:.1f} | {len(ids)} | {pct(majority_correct, len(ids))} "
              f"| {pct(cot_correct, len(ids))} | {pct(ddx_correct, len(ids))} |")
    print()


def section_paths(ddx, cot):
    print("## 2. Light vs full path\n")
    print("| path | questions | DDx-RAG | CoT on same questions |")
    print("|---|---|---|---|")
    for path in ("light", "full"):
        ids = [qid for qid, row in ddx.items() if row["trace"].get("path") == path]
        ddx_correct = sum(1 for qid in ids if ddx[qid]["correct"])
        cot_correct = sum(1 for qid in ids if qid in cot and cot[qid]["correct"])
        print(f"| {path} | {len(ids)} | {pct(ddx_correct, len(ids))} | {pct(cot_correct, len(ids))} |")
    print()


def section_adjudication(ddx):
    """changed_from_majority = the final answer differs from the closed-book majority."""
    print("## 3. Adjudication: overruling the closed-book majority\n")
    changed = [row for row in ddx.values() if row["trace"].get("changed_from_majority")]
    good = 0      # majority was wrong, DDx-RAG right
    bad = 0       # majority was right, DDx-RAG wrong
    neutral = 0   # both wrong
    for row in changed:
        votes = {k: v for k, v in row["trace"].get("votes", {}).items() if k not in ("None", "null")}
        majority = max(votes, key=lambda k: (votes[k], k)) if votes else None
        if row["correct"]:
            good += 1
        elif majority == row["gold"]:
            bad += 1
        else:
            neutral += 1
    print(f"- Overruled the majority on {len(changed)} of {len(ddx)} questions.")
    print(f"- Good overrules (majority wrong -> now right): **{good}**")
    print(f"- Bad overrules (majority right -> now wrong): **{bad}**")
    print(f"- Wrong either way: {neutral}\n")


def section_intents(ddx):
    print("## 4. Evidence in the prompt, by query intent\n")
    print("| answer | questions | " + " | ".join(["SUPPORT", "DISCRIMINATE", "REFUTE", "QUESTION"]) +
          " | uncited points dropped (avg) |")
    print("|---|---|---|---|---|---|---|")
    for label, want in (("correct", True), ("wrong", False)):
        rows = [row for row in ddx.values() if row["correct"] == want]
        counts = Counter()
        dropped = 0
        for row in rows:
            for item in row["trace"].get("evidence", []):
                for intent in set(item.get("intents", [])):
                    counts[intent] += 1
            verification = row["trace"].get("verification") or {}
            dropped += verification.get("uncited_points_dropped", 0)
        n = max(1, len(rows))
        cells = " | ".join(f"{counts[i] / n:.2f}" for i in ("SUPPORT", "DISCRIMINATE", "REFUTE", "QUESTION"))
        print(f"| {label} | {len(rows)} | {cells} | {dropped / n:.2f} |")
    print("\n(average number of evidence snippets per question found by each intent)\n")


def section_agent(name, rows):
    """DxAgent: what the loop did, measured against its OWN first answer (before any evidence)."""
    print(f"## 7. Agent loop: `{name}`\n")
    first_right = first_wrong = harm = recovery = 0
    new_hypothesis_wins = fallback = critic_changed = open_issue_rows = open_issue_right = 0
    turns = critiques = 0
    for row in rows.values():
        trace = row["trace"]
        turns += trace.get("turns", 0)
        critiques += trace.get("critiques", 0)
        fallback += bool(trace.get("mapping_fallback"))

        first_correct = trace.get("first", {}).get("choice") == row["gold"]
        if first_correct:
            first_right += 1
            if not row["correct"]:
                harm += 1                           # the loop broke a right first answer
        else:
            first_wrong += 1
            if row["correct"]:
                recovery += 1                       # the loop fixed a wrong first answer

        # Did the final answer come from a hypothesis that was NOT in the first differential?
        final = (trace.get("final", {}).get("hypothesis") or "").lower()
        for hypothesis_name, h in trace.get("hypotheses", {}).items():
            if hypothesis_name.lower() == final and h.get("origin") != "frame" and row["correct"]:
                new_hypothesis_wins += 1

        for event in trace.get("events", []):
            if event.get("event") == "critique" and (event.get("blocking") or event.get("new_hypotheses")):
                critic_changed += 1
                break

        if trace.get("open_issues"):
            open_issue_rows += 1
            open_issue_right += row["correct"]

    n = max(1, len(rows))
    print(f"- First answer (options-free, before any evidence) correct: {pct(first_right, len(rows))}")
    print(f"- Harm: right first answer ended wrong: **{harm}** of {first_right}")
    print(f"- Recovery: wrong first answer ended right: **{recovery}** of {first_wrong}")
    print(f"- Right answers that came from a hypothesis added during the loop: {new_hypothesis_wins}")
    print(f"- Critic raised a blocking issue or a new diagnosis: {critic_changed} questions")
    print(f"- Questions ending with an open issue: {open_issue_rows}, accuracy on them {pct(open_issue_right, open_issue_rows)}")
    print(f"- Mapping fallbacks (no option matched the free-text answer): {fallback}")
    print(f"- Average turns {turns / n:.2f}, critiques {critiques / n:.2f}\n")


def section_subjects(runs, ids):
    """MMLU ids look like 'anatomy-000', 'college_medicine-012'."""
    subjects = defaultdict(list)
    for qid in ids:
        match = re.match(r"([a-z_]+)-\d+$", qid)
        if match:
            subjects[match.group(1)].append(qid)
    if len(subjects) < 2:
        return
    print("## 5. Accuracy per subject\n")
    names = list(runs)
    print("| subject | n | " + " | ".join(names) + " |")
    print("|---|---|" + "---|" * len(names))
    for subject in sorted(subjects):
        sids = subjects[subject]
        cells = []
        for name in names:
            correct = sum(1 for qid in sids if runs[name][qid]["correct"])
            cells.append(f"{100 * correct / len(sids):.1f}")
        print(f"| {subject} | {len(sids)} | " + " | ".join(cells) + " |")
    print()


def section_fixed_broke(runs, ids, baseline, out_path):
    print(f"## 6. Questions each method fixed / broke relative to `{baseline}`\n")
    lists = {}
    for name, rows in runs.items():
        if name == baseline:
            continue
        fixed = [qid for qid in ids if rows[qid]["correct"] and not runs[baseline][qid]["correct"]]
        broke = [qid for qid in ids if not rows[qid]["correct"] and runs[baseline][qid]["correct"]]
        lists[name] = {"fixed": fixed, "broke": broke}
        print(f"- **{name}**: fixed {len(fixed)}, broke {len(broke)}. First broken: {', '.join(broke[:8]) or '-'}")
    with open(out_path, "w") as f:
        json.dump(lists, f, indent=1)
    print(f"\nFull lists: `{out_path}`\n")


def main():
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="medqa")
    parser.add_argument("--run-name", default=settings.llm_model.replace("/", "_"))
    parser.add_argument("--baseline", default="cot")
    parser.add_argument("--ddx", default="ddx_rag", help="which DDx-RAG result file to dissect")
    args = parser.parse_args()

    run_dir = os.path.join("results", args.run_name, args.dataset)
    runs = load_run(run_dir)
    if args.baseline not in runs:
        sys.exit(f"No {args.baseline}.jsonl in {run_dir}")

    common = None
    for rows in runs.values():
        common = set(rows) if common is None else common & set(rows)
    ids = sorted(common)

    print(f"# Error analysis: {args.dataset}, {args.run_name} ({len(ids)} questions answered by all methods)\n")
    if args.ddx in runs:
        ddx = {qid: runs[args.ddx][qid] for qid in ids}
        cot = runs[args.baseline]
        section_gate_signal(ddx, cot)
        section_paths(ddx, cot)
        section_adjudication(ddx)
        section_intents(ddx)
    for name, rows in runs.items():
        if name.startswith("dx_agent"):
            section_agent(name, {qid: rows[qid] for qid in ids})
    section_subjects(runs, ids)
    section_fixed_broke(runs, ids, args.baseline, os.path.join(run_dir, "fixed_broke.json"))


if __name__ == "__main__":
    main()
