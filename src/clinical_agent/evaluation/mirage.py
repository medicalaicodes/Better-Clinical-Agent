"""
MIRAGE benchmark runner (Xiong et al., ACL Findings 2024; github.com/Teddy-XiongGZ/MIRAGE).

Datasets: medqa (1,273), mmlu (1,089), medmcqa (4,183), pubmedqa (500), bioasq (618).

We also provide "medqa_dev": 300 questions sampled from the MedQA *training* set. All
thresholds are tuned there, so the test numbers are never tuned on the test questions.

Results are written as one JSON line per question, appended the moment the question is
done. If a run crashes or is stopped, rerunning the same command continues where it stopped.
"""

import json
import os
import random
import threading
from typing import Callable, Dict, List, Optional

import requests

from ..rag import MCQ
from .stats import help_harm, mcnemar_exact, wilson_interval

MIRAGE_URL = "https://raw.githubusercontent.com/Teddy-XiongGZ/MIRAGE/main/benchmark.json"
MEDQA_TRAIN_URL = ("https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options/resolve/main/"
                   "phrases_no_exclude_train.jsonl")
DATASETS = ("medqa", "mmlu", "medmcqa", "pubmedqa", "bioasq", "medqa_dev")


def download_once(url: str, path: str) -> str:
    if os.path.exists(path):
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    with open(path, "wb") as f:
        f.write(response.content)
    return path


# ----------------------------------------------------------------------------------------
# Loading questions
# ----------------------------------------------------------------------------------------

def load_questions(dataset: str, data_dir: str, limit: Optional[int] = None, seed: int = 0,
                   shuffle_options: Optional[int] = None) -> List[MCQ]:
    """Loads a MIRAGE dataset (or the MedQA dev split) as a list of MCQ.

    limit            only the first N questions (medqa_dev: first N of its fixed 300-question sample)
    shuffle_options  robustness test: reorder every question's options with this seed
    """
    benchmarks_dir = os.path.join(data_dir, "benchmarks")

    if dataset == "medqa_dev":
        path = download_once(MEDQA_TRAIN_URL, os.path.join(benchmarks_dir, "medqa_train.jsonl"))
        with open(path, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f]

        # Always draw the same sample (300 questions, or more if asked), then cut to `limit`.
        # Sampling `limit` questions directly would give DIFFERENT questions with the same ids
        # (train-0, train-1, ...) and silently mix them up with earlier result files.
        sample_size = min(max(limit or 0, 300), len(rows))
        rows = random.Random(seed).sample(rows, sample_size)
        if limit:
            rows = rows[:limit]

        questions = []
        for number, row in enumerate(rows):
            questions.append(MCQ(id=f"train-{number}", question=row["question"], options=row["options"],
                                 answer=row["answer_idx"], dataset=dataset))
    else:
        path = download_once(MIRAGE_URL, os.path.join(benchmarks_dir, "mirage_benchmark.json"))
        with open(path, encoding="utf-8") as f:
            data = json.load(f)[dataset]

        questions = []
        for question_id, item in data.items():
            questions.append(MCQ(id=question_id, question=item["question"], options=item["options"],
                                 answer=item["answer"], dataset=dataset))
        if limit:
            questions = questions[:limit]

    if shuffle_options is not None:
        shuffled = []
        for q in questions:
            # Seeded per question, so the same question is shuffled the same way every run.
            rng = random.Random(f"{shuffle_options}-{q.id}")
            shuffled.append(shuffle_options_of(q, rng))
        questions = shuffled

    return questions


def shuffle_options_of(q: MCQ, rng: random.Random) -> MCQ:
    """Same question, options in a new order, gold letter moved along with its text."""
    letters = sorted(q.options)                         # ["A", "B", "C", "D"]
    texts = [q.options[letter] for letter in letters]

    order = list(range(len(texts)))                     # order[new position] = old position
    rng.shuffle(order)

    new_options = {}
    for new_position, old_position in enumerate(order):
        new_options[letters[new_position]] = texts[old_position]

    old_gold_position = letters.index(q.answer)
    new_gold_letter = letters[order.index(old_gold_position)]
    return MCQ(q.id, q.question, new_options, new_gold_letter, q.dataset)


# ----------------------------------------------------------------------------------------
# Running a method
# ----------------------------------------------------------------------------------------

def read_results(path: str) -> Dict[str, dict]:
    """question id -> result row, for every line already written to `path`."""
    if not os.path.exists(path):
        return {}
    results = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                results[row["id"]] = row
    return results


def run_method(method: Callable, questions: List[MCQ], out_path: str, llm, workers: int = 8) -> Dict[str, dict]:
    """Answers every question that is not already in `out_path`, `workers` at a time."""
    from tqdm import tqdm
    from ..llm import parallel_map

    done = read_results(out_path)
    todo = [q for q in questions if q.id not in done]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    write_lock = threading.Lock()

    def solve(q: MCQ) -> dict:
        # Runs inside a worker thread: usage() therefore counts this question's calls only.
        llm.reset_usage()
        try:
            result = method(q)
            error = None
        except Exception as exc:
            # One broken question must not kill a two-hour run. The error is saved with the
            # row and the question counts as wrong.
            result = {"answer": None, "trace": {}}
            error = f"{type(exc).__name__}: {exc}"

        return {
            "id": q.id,
            "gold": q.answer,
            "pred": result["answer"],
            "correct": result["answer"] == q.answer,
            "usage": llm.usage(),
            "error": error,
            "trace": result["trace"],
        }

    with open(out_path, "a", encoding="utf-8") as f:
        progress = tqdm(parallel_map(solve, todo, workers), total=len(todo), desc=os.path.basename(out_path))
        for q, row in progress:
            with write_lock:
                f.write(json.dumps(row) + "\n")
                f.flush()          # flushed per line, so a crash loses at most one question
            done[q.id] = row
    return done


# ----------------------------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------------------------

def summarize(results: Dict[str, dict], ids: List[str]) -> dict:
    rows = [results[i] for i in ids if i in results]
    n = len(rows)
    if n == 0:
        return {"n": 0, "accuracy": 0.0, "ci95": (0.0, 0.0), "unparsed": 0, "errors": 0,
                "calls_per_q": 0.0, "tokens_per_q": 0.0}

    correct = sum(1 for r in rows if r["correct"])
    total_calls = sum(r["usage"]["calls"] for r in rows)
    total_tokens = sum(r["usage"]["prompt_tokens"] + r["usage"]["completion_tokens"] for r in rows)

    return {
        "n": n,
        "accuracy": correct / n,
        "ci95": wilson_interval(correct, n),
        "unparsed": sum(1 for r in rows if r["pred"] is None),
        "errors": sum(1 for r in rows if r.get("error")),
        "calls_per_q": total_calls / n,
        "tokens_per_q": total_tokens / n,
    }


def comparison_table(results_by_method: Dict[str, Dict[str, dict]], baseline: str) -> str:
    """Markdown table over the questions EVERY method has answered (so rows are comparable):
    accuracy, 95% CI, paired McNemar test vs the baseline, fixed/broke counts, and cost."""
    if not results_by_method:
        return ""

    common_ids = None
    for results in results_by_method.values():
        ids = set(results)
        common_ids = ids if common_ids is None else common_ids & ids
    ids = sorted(common_ids)

    lines = [
        f"| method | n | accuracy | 95% CI | vs {baseline} (p) | fixed / broke | unparsed | errors | calls/q | tokens/q |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]

    base_results = results_by_method.get(baseline)
    base_summary = summarize(base_results, ids) if base_results else None

    for name, results in results_by_method.items():
        s = summarize(results, ids)

        versus = "-"
        fixed_broke = "-"
        if base_results is not None and name != baseline:
            base_correct = [base_results[i]["correct"] for i in ids]
            this_correct = [results[i]["correct"] for i in ids]
            test = mcnemar_exact(base_correct, this_correct)
            counts = help_harm(base_correct, this_correct)
            difference = 100 * (s["accuracy"] - base_summary["accuracy"])
            versus = f"{difference:+.1f} (p={test['p_value']:.3f})"
            fixed_broke = f"{counts['fixed']} / {counts['broke']}"

        low, high = s["ci95"]
        lines.append(f"| {name} | {s['n']} | {100 * s['accuracy']:.1f} | {100 * low:.1f}-{100 * high:.1f} "
                     f"| {versus} | {fixed_broke} | {s['unparsed']} | {s['errors']} "
                     f"| {s['calls_per_q']:.1f} | {s['tokens_per_q']:.0f} |")
    return "\n".join(lines)
