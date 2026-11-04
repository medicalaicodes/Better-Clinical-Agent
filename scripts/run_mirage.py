"""
Run medical QA methods on MIRAGE and print a comparison table.

    # all main methods on MedQA (needs the vLLM server running, see scripts/serve_vllm.sh)
    python scripts/run_mirage.py --dataset medqa --methods cot cot_sc medrag rationale_rag imedrag ddx_rag

    # an ablation: DDx-RAG without REFUTE queries, saved under its own tag
    python scripts/run_mirage.py --dataset medqa --methods ddx_rag --set intents=SUPPORT,DISCRIMINATE --tag no_refute

    # robustness: shuffled options / 3 random snippets injected into every retrieval
    python scripts/run_mirage.py --dataset medqa --methods ddx_rag --shuffle-options 1 --tag shuffled
    python scripts/run_mirage.py --dataset medqa --methods ddx_rag --noise 3 --tag noise3

    # just print the table for what has already been run
    python scripts/run_mirage.py --dataset medqa --report-only

Results: results/<run-name>/<dataset>/<method>[__<tag>].jsonl   (one line per question, resumable)
"""

import argparse
import glob
import inspect
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.clinical_agent.config import get_settings  # noqa: E402
from src.clinical_agent.evaluation.mirage import (DATASETS, comparison_table, load_questions,  # noqa: E402
                                                  read_results, run_method)
from src.clinical_agent.llm import LLM  # noqa: E402
from src.clinical_agent.rag import METHODS, RETRIEVAL_FREE  # noqa: E402


def parse_overrides(pairs):
    """--set key=value pairs -> dict with ints/floats/bools converted."""
    overrides = {}
    for pair in pairs or []:
        key, value = pair.split("=", 1)
        for cast in (int, float):
            try:
                value = cast(value)
                break
            except ValueError:
                continue
        if value in ("true", "false", "True", "False"):
            value = value.lower() == "true"
        overrides[key] = value
    return overrides


def main():
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=DATASETS, default="medqa")
    parser.add_argument("--methods", nargs="+", default=["cot", "medrag", "ddx_rag"], choices=sorted(METHODS))
    parser.add_argument("--limit", type=int, help="first N questions only")
    parser.add_argument("--workers", type=int, default=48, help="questions in flight at once (vLLM batches them)")
    parser.add_argument("--set", nargs="*", dest="overrides", help="method settings, e.g. gate_threshold=0.6")
    parser.add_argument("--tag", default="", help="suffix for ablation / robustness runs")
    parser.add_argument("--run-name", default=None, help="results sub-folder (default: LLM model name)")
    parser.add_argument("--shuffle-options", type=int, help="robustness: permute options with this seed")
    parser.add_argument("--noise", type=int, default=0, help="robustness: random snippets added per retrieval")
    parser.add_argument("--baseline", default="cot", help="method the table compares against")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--device", default=settings.device)
    args = parser.parse_args()

    run_dir = os.path.join("results", args.run_name or settings.llm_model.replace("/", "_"), args.dataset)
    questions = load_questions(args.dataset, settings.data_dir, limit=args.limit,
                               shuffle_options=args.shuffle_options)

    if not args.report_only:
        llm = LLM()
        retriever = None
        if any(m not in RETRIEVAL_FREE for m in args.methods):
            from src.clinical_agent.retrieval import Retriever
            retriever = Retriever.load(settings.index_dir, device=args.device)
            retriever.noise_snippets = args.noise

        overrides = parse_overrides(args.overrides)
        os.makedirs(run_dir, exist_ok=True)
        for name in args.methods:
            # each method only receives the --set keys it actually has
            accepted = inspect.signature(METHODS[name].__init__).parameters
            method = METHODS[name](llm, retriever, **{k: v for k, v in overrides.items() if k in accepted})
            out_path = os.path.join(run_dir, f"{name}{'__' + args.tag if args.tag else ''}.jsonl")
            with open(out_path.replace(".jsonl", ".config.json"), "w") as f:
                json.dump({"model": settings.llm_model, "method": name, "overrides": overrides,
                           "dataset": args.dataset, "limit": args.limit, "noise": args.noise,
                           "shuffle_options": args.shuffle_options}, f, indent=2)
            print(f"\n== {name}{' [' + args.tag + ']' if args.tag else ''} on {args.dataset} ({len(questions)} questions)")
            run_method(method, questions, out_path, llm, workers=args.workers)

    # Table over every result file in this run folder, restricted to the loaded questions.
    ids = {q.id for q in questions}
    results = {}
    for path in sorted(glob.glob(os.path.join(run_dir, "*.jsonl"))):
        name = os.path.basename(path)[:-len(".jsonl")]
        rows = {i: r for i, r in read_results(path).items() if i in ids}
        if rows:
            results[name] = rows
    if results:
        print(f"\n### {args.dataset} - {settings.llm_model} (questions answered by every method)\n")
        print(comparison_table(results, baseline=args.baseline if args.baseline in results else next(iter(results))))


if __name__ == "__main__":
    main()
