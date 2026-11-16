"""
Runs medical QA methods on MIRAGE and prints a comparison table.

    # the baselines on MedQA (needs the vLLM server running, see scripts/serve_vllm.sh)
    python scripts/run_mirage.py --dataset medqa --methods cot cot_sc medrag rationale_rag imedrag

    # the agent on 20 practice questions, with its settings changed through --set
    python scripts/run_mirage.py --dataset medqa_dev --methods dx_agent --set final=select --limit 20

    # only print the table for what has already been run
    python scripts/run_mirage.py --dataset medqa --report-only

Results go to results/<run-name>/<dataset>/<method>[__<tag>].jsonl - one line per question.
Running the same command again continues where it stopped.
"""

import argparse
import glob
import json
import os
import sys

# Lets this script import the project's code when it's run from the repository folder.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.clinical_agent.config import get_settings  # noqa: E402
from src.clinical_agent.evaluation.mirage import (DATASETS, comparison_table, load_questions,  # noqa: E402
                                                  read_results, run_method)
from src.clinical_agent.llm import LLM  # noqa: E402
from src.clinical_agent.rag import METHODS, RETRIEVAL_FREE  # noqa: E402


def read_settings(pairs):
    """["final=select", "max_rounds=0", "library=false"]  ->  {"final": "select", "max_rounds": 0, "library": False}"""
    settings = {}
    for pair in pairs or []:
        key, value = pair.split("=", 1)
        if value.lower() == "true":
            value = True
        elif value.lower() == "false":
            value = False
        else:
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    pass           # not a number: keep the text as it is
        settings[key] = value
    return settings


def main():
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=DATASETS, default="medqa")
    parser.add_argument("--methods", nargs="+", default=["cot"], choices=sorted(METHODS))
    parser.add_argument("--limit", type=int, help="only the first N questions")
    parser.add_argument("--workers", type=int, default=48, help="questions answered at the same time")
    parser.add_argument("--set", nargs="*", dest="method_settings", help="method settings, e.g. max_rounds=2")
    parser.add_argument("--tag", default="", help="added to the file name, to keep variants apart")
    parser.add_argument("--run-name", default=None, help="results sub-folder (default: the model name)")
    parser.add_argument("--baseline", default="cot", help="the method the table compares against")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--device", default=settings.device)
    args = parser.parse_args()

    run_name = args.run_name or settings.llm_model.replace("/", "_")
    run_dir = os.path.join("results", run_name, args.dataset)
    questions = load_questions(args.dataset, settings.data_dir, limit=args.limit)

    if not args.report_only:
        llm = LLM()

        # The big index (~3 GB) is only loaded if some method actually searches the library.
        retriever = None
        needs_library = False
        for name in args.methods:
            if name not in RETRIEVAL_FREE:
                needs_library = True
        if needs_library:
            import torch
            from src.clinical_agent.retrieval import load_retriever
            if args.device is None and not torch.cuda.is_available():
                print("WARNING: no GPU visible to torch - MedCPT retrieval will run on CPU (slow).")
            retriever = load_retriever(settings.index_dir, device=args.device)

        method_settings = read_settings(args.method_settings)
        os.makedirs(run_dir, exist_ok=True)

        for name in args.methods:
            method = METHODS[name](llm, retriever, **method_settings)

            file_name = name
            if args.tag:
                file_name = name + "__" + args.tag
            out_path = os.path.join(run_dir, file_name + ".jsonl")

            # Save how this run was made, next to its results.
            with open(os.path.join(run_dir, file_name + ".config.json"), "w") as f:
                json.dump({"model": settings.llm_model, "method": name, "settings": method_settings,
                           "dataset": args.dataset, "limit": args.limit}, f, indent=2)

            print(f"\n== {file_name} on {args.dataset} ({len(questions)} questions)")
            run_method(method, questions, out_path, llm, workers=args.workers)

    # The table covers every result file in this folder, for the questions loaded above.
    wanted_ids = set()
    for q in questions:
        wanted_ids.add(q.id)

    results = {}
    for path in sorted(glob.glob(os.path.join(run_dir, "*.jsonl"))):
        name = os.path.basename(path)[:-len(".jsonl")]
        rows = {}
        for question_id, row in read_results(path).items():
            if question_id in wanted_ids and "correct" in row:      # skip files that aren't method results
                rows[question_id] = row
        if rows:
            results[name] = rows

    if results:
        baseline = args.baseline
        if baseline not in results:
            baseline = list(results)[0]
        print(f"\n### {args.dataset} - {settings.llm_model} (questions answered by every method)\n")
        print(comparison_table(results, baseline))


if __name__ == "__main__":
    main()
