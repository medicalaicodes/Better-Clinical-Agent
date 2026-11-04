"""
Go / no-go probe for book scoring (prior x textbook likelihood), see
src/clinical_agent/rag/book_scoring.py for the idea.

Step 1 - features (GPU, no LLM needed; reuses the saved cot_sc votes):
    python scripts/probe_book_scoring.py features --dataset medqa_dev
    python scripts/probe_book_scoring.py features --dataset medqa
    python scripts/probe_book_scoring.py features --dataset mmlu

Step 2 - fit on the tuning questions, evaluate once on the test sets (CPU, seconds):
    python scripts/probe_book_scoring.py fit --train medqa_dev --eval medqa mmlu

Protocol (so the numbers are honest):
  - Every weight is fitted on MedQA TRAINING questions (medqa_dev) only.
  - The feature set is chosen by 5-fold cross-validation on those same training questions,
    BEFORE looking at the test sets. The other feature sets are printed for transparency.
  - The bar is cot_sc (majority of the same 5 votes): the probe adds zero LLM calls.

Note: this probe uses the answer OPTIONS as candidates. That is only to test whether
"textbook fit" can pick the right candidate at all; the real system will be options-free.
"""

import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.clinical_agent.config import get_settings  # noqa: E402
from src.clinical_agent.evaluation.mirage import load_questions, read_results  # noqa: E402
from src.clinical_agent.evaluation.stats import mcnemar_exact, wilson_interval  # noqa: E402
from src.clinical_agent.rag import book_scoring as bs  # noqa: E402

FEATURE_FILE = "book_features.jsonl"


# ----------------------------------------------------------------------------------------
# Step 1: features
# ----------------------------------------------------------------------------------------

def run_features(args, settings):
    from tqdm import tqdm
    from src.clinical_agent.retrieval import Retriever

    run_dir = os.path.join("results", args.run_name, args.dataset)
    votes_by_id = read_results(os.path.join(run_dir, f"{args.votes}.jsonl"))
    if not votes_by_id:
        sys.exit(f"No {args.votes}.jsonl in {run_dir}: run cot_sc on {args.dataset} first.")

    out_path = os.path.join(run_dir, FEATURE_FILE)
    done = read_results(out_path)                       # resumable, like run_mirage.py

    # Load the FULL dataset and cut afterwards: for medqa_dev, `limit` would change the random
    # sample itself, and the question ids would no longer match the saved votes.
    questions = load_questions(args.dataset, settings.data_dir)
    if args.limit:
        questions = questions[:args.limit]
    todo = [q for q in questions if q.id in votes_by_id and q.id not in done]
    print(f"{args.dataset}: {len(todo)} questions to score ({len(done)} already done)")

    retriever = Retriever.load(settings.index_dir, device=args.device)
    scorer = bs.BookScorer(retriever, passages_per_candidate=args.k)

    with open(out_path, "a", encoding="utf-8") as f:
        for q in tqdm(todo, desc="book scoring"):
            row = {
                "id": q.id,
                "gold": q.answer,
                "qtype": bs.question_type(q.question),
                "votes": votes_by_id[q.id]["trace"]["votes"],
                "book": scorer.score_question(q.question, q.options),
            }
            f.write(json.dumps(row) + "\n")
            f.flush()
    print(f"Wrote {out_path}")


# ----------------------------------------------------------------------------------------
# Step 2: fit + report
# ----------------------------------------------------------------------------------------

def pct(correct, total):
    low, high = wilson_interval(correct, total)
    return f"{100 * correct / total:.1f} ({100 * low:.1f}-{100 * high:.1f})"


def agreement_of(votes):
    valid = [v for k, v in votes.items() if k != "None"]
    return max(valid) / sum(valid) if valid else 0.0


def report_dataset(name, rows, fitted, chosen):
    """Prints one test set's table. `fitted` = feature-set name -> predict function."""
    print(f"\n## {name} ({len(rows)} questions)\n")
    sc_letters = [bs.majority_letter(r["votes"]) for r in rows]
    sc_correct = [letter == r["gold"] for letter, r in zip(sc_letters, rows)]

    print("| method | accuracy (95% CI) | vs cot_sc | fixed / broke |")
    print("|---|---|---|---|")
    print(f"| cot_sc (majority of 5, the bar) | {pct(sum(sc_correct), len(rows))} | - | - |")

    per_method_correct = {}
    for set_name, predict in fitted.items():
        columns = bs.FEATURE_SETS[set_name.replace(" (mlp)", "")]
        questions = bs.build_matrix(rows, columns)
        letters = bs.predict_letters(predict, questions)
        correct = [letter == r["gold"] for letter, r in zip(letters, rows)]
        per_method_correct[set_name] = correct
        test = mcnemar_exact(sc_correct, correct)
        fixed = sum(1 for a, b in zip(sc_correct, correct) if b and not a)
        broke = sum(1 for a, b in zip(sc_correct, correct) if a and not b)
        diff = 100 * (sum(correct) - sum(sc_correct)) / len(rows)
        mark = " **(pre-registered)**" if set_name == chosen else ""
        print(f"| {set_name}{mark} | {pct(sum(correct), len(rows))} | {diff:+.1f} (p={test['p_value']:.3f}) "
              f"| {fixed} / {broke} |")

    # Where does it help? By question type and by vote agreement, for the chosen set.
    chosen_correct = per_method_correct[chosen]
    for label, key_of in (("question type", lambda r: r["qtype"]),
                          ("vote agreement", lambda r: f"{agreement_of(r['votes']):.1f}")):
        groups = defaultdict(list)
        for i, r in enumerate(rows):
            groups[key_of(r)].append(i)
        print(f"\nBy {label} (`{chosen}` vs cot_sc; book-only = `book_findings`):\n")
        print(f"| {label} | n | cot_sc | {chosen} | book-only |")
        print("|---|---|---|---|---|")
        book_only = per_method_correct.get("book_findings", chosen_correct)
        for key in sorted(groups, reverse=True):
            idx = groups[key]
            def acc(values):
                return f"{100 * sum(values[i] for i in idx) / len(idx):.1f}"
            print(f"| {key} | {len(idx)} | {acc(sc_correct)} | {acc(chosen_correct)} | {acc(book_only)} |")


def run_fit(args):
    def load(dataset):
        path = os.path.join("results", args.run_name, dataset, FEATURE_FILE)
        rows = list(read_results(path).values())
        if not rows:
            sys.exit(f"No features at {path}: run the 'features' step for {dataset} first.")
        return rows

    train_rows = load(args.train)
    print(f"# Book-scoring probe ({args.run_name})\n")
    print(f"Tuning set: {args.train}, {len(train_rows)} questions. All weights and the feature-set "
          f"choice come from here only.\n")

    # --- choose the feature set by cross-validation on the tuning set ---
    print("## Cross-validated accuracy on the tuning set (5 folds)\n")
    print("| feature set | columns | CV accuracy |")
    print("|---|---|---|")
    cv = {}
    for set_name, columns in bs.FEATURE_SETS.items():
        questions = bs.build_matrix(train_rows, columns)
        cv[set_name] = bs.cross_validated_accuracy(questions, l2=args.l2)
        print(f"| {set_name} | {', '.join(columns)} | {100 * cv[set_name]:.1f} |")
    full_questions = bs.build_matrix(train_rows, bs.FEATURE_SETS["full"])
    mlp_cv = bs.cross_validated_accuracy(full_questions, l2=args.l2, hidden=8)
    print(f"| full (mlp) | same as full, 8 hidden units | {100 * mlp_cv:.1f} |")

    # Pre-register: the best CV set among those that use the book (ties -> fewer columns).
    candidates = [s for s in bs.FEATURE_SETS if s.startswith("votes+") or s == "full"]
    chosen = max(candidates, key=lambda s: (round(cv[s], 4), -len(bs.FEATURE_SETS[s])))
    print(f"\nPre-registered choice (decided before any test set is read): **{chosen}**\n")

    # --- fit every set on the whole tuning set ---
    fitted = {}
    for set_name, columns in bs.FEATURE_SETS.items():
        fitted[set_name] = bs.fit_conditional_logit(bs.build_matrix(train_rows, columns), l2=args.l2)
    fitted["full (mlp)"] = bs.fit_conditional_logit(full_questions, l2=args.l2, hidden=8)

    print("Learned weights (per unit of each feature) for the linear sets:\n")
    for set_name in bs.FEATURE_SETS:
        weights = fitted[set_name].weights
        pairs = ", ".join(f"{c}={w:+.2f}" for c, w in zip(bs.FEATURE_SETS[set_name], weights))
        print(f"- {set_name}: {pairs}")

    for dataset in args.eval:
        report_dataset(dataset, load(dataset), fitted, chosen)


def main():
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="step", required=True)

    features = sub.add_parser("features", help="score every candidate against the textbooks (GPU)")
    features.add_argument("--dataset", required=True)
    features.add_argument("--votes", default="cot_sc", help="result file holding the 5 closed-book votes")
    features.add_argument("--k", type=int, default=8, help="passages per candidate")
    features.add_argument("--limit", type=int)
    features.add_argument("--device", default=settings.device)

    fit = sub.add_parser("fit", help="fit the combiner on --train, evaluate on --eval (CPU)")
    fit.add_argument("--train", default="medqa_dev")
    fit.add_argument("--eval", nargs="+", default=["medqa", "mmlu"])
    fit.add_argument("--l2", type=float, default=0.01)

    for p in (features, fit):
        p.add_argument("--run-name", default=settings.llm_model.replace("/", "_"))
    args = parser.parse_args()

    if args.step == "features":
        run_features(args, settings)
    else:
        run_fit(args)


if __name__ == "__main__":
    main()
