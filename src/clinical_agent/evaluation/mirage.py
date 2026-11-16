"""
Runs methods on the MIRAGE benchmark (Xiong et al., ACL Findings 2024).

Datasets: medqa (1,273 questions), mmlu (1,089), medmcqa (4,183), pubmedqa (500), bioasq (618).

"medqa_dev" is extra: 300 questions sampled from the MedQA TRAINING set. Anything we tune is
tuned there, so the test numbers are never tuned on the test questions.

Results are saved as one JSON line per question, written the moment the question is done.
If a run crashes or is stopped, running the same command again continues where it stopped.
"""

import json
import os
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

from ..rag import MCQ
from .stats import mcnemar_exact, wilson_interval

MIRAGE_URL = "https://raw.githubusercontent.com/Teddy-XiongGZ/MIRAGE/main/benchmark.json"
MEDQA_TRAIN_URL = ("https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options/resolve/main/"
                   "phrases_no_exclude_train.jsonl")
DATASETS = ["medqa", "mmlu", "medmcqa", "pubmedqa", "bioasq", "medqa_dev"]


def download_once(url, path):
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

def load_questions(dataset, data_dir, limit=None, seed=0):
    """A list of MCQ questions. `limit` keeps only the first N."""
    benchmarks_dir = os.path.join(data_dir, "benchmarks")
    questions = []

    if dataset == "medqa_dev":
        path = download_once(MEDQA_TRAIN_URL, os.path.join(benchmarks_dir, "medqa_train.jsonl"))
        rows = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                rows.append(json.loads(line))

        # Always draw the same 300 questions, then cut to `limit`. Drawing `limit` questions
        # directly would give DIFFERENT questions under the same ids (train-0, train-1, ...)
        # and silently mix them up with older result files.
        sample_size = min(max(limit or 0, 300), len(rows))
        rows = random.Random(seed).sample(rows, sample_size)
        if limit:
            rows = rows[:limit]

        number = 0
        for row in rows:
            questions.append(MCQ(f"train-{number}", row["question"], row["options"], row["answer_idx"], dataset))
            number += 1
    else:
        path = download_once(MIRAGE_URL, os.path.join(benchmarks_dir, "mirage_benchmark.json"))
        with open(path, encoding="utf-8") as f:
            data = json.load(f)[dataset]

        for question_id in data:
            item = data[question_id]
            questions.append(MCQ(question_id, item["question"], item["options"], item["answer"], dataset))
        if limit:
            questions = questions[:limit]

    return questions


# ----------------------------------------------------------------------------------------
# Running a method
# ----------------------------------------------------------------------------------------

def read_results(path):
    """{question id: result row} for every line already written to `path`."""
    results = {}
    if not os.path.exists(path):
        return results
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                results[row["id"]] = row
    return results


def run_method(method, questions, out_path, llm, workers=8):
    """Answers every question that isn't in `out_path` yet, `workers` questions at a time."""
    from tqdm import tqdm

    done = read_results(out_path)
    todo = []
    for q in questions:
        if q.id not in done:
            todo.append(q)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    write_lock = threading.Lock()

    def solve(q):
        # This runs inside one worker thread, so llm.usage() counts only this question's calls.
        llm.reset_usage()
        try:
            result = method(q)
            error = None
        except Exception as problem:
            # One broken question must not kill a two-hour run: we save the error with the row
            # and the question simply counts as wrong.
            result = {"answer": None, "trace": {}}
            error = type(problem).__name__ + ": " + str(problem)

        return {
            "id": q.id,
            "gold": q.answer,
            "pred": result["answer"],
            "correct": result["answer"] == q.answer,
            "usage": llm.usage(),
            "error": error,
            "trace": result["trace"],
        }

    # Threads are enough here: each one mostly waits for the LLM server, and vLLM answers all
    # the waiting requests together on the GPU.
    with open(out_path, "a", encoding="utf-8") as f:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futures = []
            for q in todo:
                futures.append(pool.submit(solve, q))

            for future in tqdm(as_completed(futures), total=len(futures), desc=os.path.basename(out_path)):
                row = future.result()
                with write_lock:
                    f.write(json.dumps(row) + "\n")
                    f.flush()          # written straight away, so a crash loses at most one question
                done[row["id"]] = row
    return done


# ----------------------------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------------------------

def summarize(results, ids):
    rows = []
    for question_id in ids:
        if question_id in results:
            rows.append(results[question_id])
    n = len(rows)
    if n == 0:
        return {"n": 0, "accuracy": 0.0, "ci95": (0.0, 0.0), "unparsed": 0, "errors": 0,
                "calls_per_q": 0.0, "tokens_per_q": 0.0}

    correct = 0
    unparsed = 0
    errors = 0
    calls = 0
    tokens = 0
    for row in rows:
        if row["correct"]:
            correct += 1
        if row["pred"] is None:
            unparsed += 1
        if row.get("error"):
            errors += 1
        calls += row["usage"]["calls"]
        tokens += row["usage"]["prompt_tokens"] + row["usage"]["completion_tokens"]

    return {
        "n": n,
        "accuracy": correct / n,
        "ci95": wilson_interval(correct, n),
        "unparsed": unparsed,
        "errors": errors,
        "calls_per_q": calls / n,
        "tokens_per_q": tokens / n,
    }


def comparison_table(results_by_method, baseline):
    """A markdown table over the questions EVERY method answered, so the rows are comparable:
    accuracy, 95% interval, paired McNemar test against the baseline, fixed/broke, and cost."""
    if not results_by_method:
        return ""

    common_ids = None
    for results in results_by_method.values():
        if common_ids is None:
            common_ids = set(results)
        else:
            common_ids = common_ids & set(results)
    ids = sorted(common_ids)

    lines = [
        f"| method | n | accuracy | 95% CI | vs {baseline} (p) | fixed / broke | unparsed | errors | calls/q | tokens/q |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]

    base_results = results_by_method.get(baseline)
    base_summary = None
    if base_results is not None:
        base_summary = summarize(base_results, ids)

    for name in results_by_method:
        results = results_by_method[name]
        s = summarize(results, ids)

        versus = "-"
        fixed_broke = "-"
        if base_results is not None and name != baseline:
            base_correct = []
            this_correct = []
            for question_id in ids:
                base_correct.append(base_results[question_id]["correct"])
                this_correct.append(results[question_id]["correct"])
            test = mcnemar_exact(base_correct, this_correct)
            difference = 100 * (s["accuracy"] - base_summary["accuracy"])
            versus = f"{difference:+.1f} (p={test['p_value']:.3f})"
            fixed_broke = f"{test['b01']} / {test['b10']}"     # fixed = it was wrong, now right

        low, high = s["ci95"]
        lines.append(f"| {name} | {s['n']} | {100 * s['accuracy']:.1f} | {100 * low:.1f}-{100 * high:.1f} "
                     f"| {versus} | {fixed_broke} | {s['unparsed']} | {s['errors']} "
                     f"| {s['calls_per_q']:.1f} | {s['tokens_per_q']:.0f} |")
    return "\n".join(lines)
