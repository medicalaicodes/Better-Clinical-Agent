"""
One small client for every LLM call in the project (the agent AND the RAG benchmarks).

It talks to any server that speaks the OpenAI chat API:
    - vLLM on the GPU server (what the benchmarks use)
    - Ollama on a laptop          (LLM_BASE_URL=http://localhost:11434/v1)
    - llama.cpp server             (what the CPU smoke test uses)

It also keeps a per-thread count of calls and tokens. The benchmark runner answers each
question inside one worker thread, so after a question finishes, `usage()` tells us exactly
how much that question cost.
"""

import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Union

from openai import OpenAI

from .config import get_settings


class LLM:
    def __init__(self, model: Optional[str] = None, base_url: Optional[str] = None,
                 api_key: Optional[str] = None, temperature: Optional[float] = None,
                 max_tokens: Optional[int] = None):
        settings = get_settings()

        # Anything not passed in explicitly comes from .env / environment variables.
        self.model = model or settings.llm_model
        self.max_tokens = max_tokens or settings.llm_max_tokens
        if temperature is None:
            self.temperature = settings.llm_temperature
        else:
            self.temperature = temperature

        self.client = OpenAI(
            base_url=base_url or settings.llm_base_url,
            api_key=api_key or settings.llm_api_key,   # vLLM ignores it, but the SDK wants one
            timeout=settings.llm_timeout,
            max_retries=3,                             # the SDK retries timeouts / 5xx for us
        )

        # threading.local() gives every thread its own private copy of these counters.
        self._usage = threading.local()

    # ------------------------------------------------------------------------------------
    # Cost accounting (per thread)
    # ------------------------------------------------------------------------------------
    def reset_usage(self) -> None:
        self._usage.calls = 0
        self._usage.prompt_tokens = 0
        self._usage.completion_tokens = 0

    def usage(self) -> Dict[str, int]:
        return {
            "calls": getattr(self._usage, "calls", 0),
            "prompt_tokens": getattr(self._usage, "prompt_tokens", 0),
            "completion_tokens": getattr(self._usage, "completion_tokens", 0),
        }

    def _count(self, response) -> None:
        # A thread that never called reset_usage() starts from zero.
        if not hasattr(self._usage, "calls"):
            self.reset_usage()

        self._usage.calls += 1

        # Streaming responses have no usage block, so only count tokens when we get one.
        if response is not None and getattr(response, "usage", None):
            self._usage.prompt_tokens += response.usage.prompt_tokens or 0
            self._usage.completion_tokens += response.usage.completion_tokens or 0

    # ------------------------------------------------------------------------------------
    # The one method everything uses
    # ------------------------------------------------------------------------------------
    def chat(self, prompt: str, system: Optional[str] = None, temperature: Optional[float] = None,
             max_tokens: Optional[int] = None, n: int = 1, seed: Optional[int] = None,
             on_token: Optional[Callable[[str], None]] = None) -> Union[str, List[str]]:
        """Sends one user prompt (plus an optional system prompt).

        Returns the reply text, or a list of `n` independent samples when n > 1.
        `on_token` streams the reply token by token (used by the interactive demo).
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        if temperature is None:
            temperature = self.temperature

        request = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens or self.max_tokens,
        }
        if seed is not None:
            request["seed"] = seed      # makes sampled answers reproducible on vLLM

        # --- streaming path (demo only) ---
        if on_token is not None:
            pieces = []
            for chunk in self.client.chat.completions.create(stream=True, **request):
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta.content
                if delta:
                    pieces.append(delta)
                    on_token(delta)
            self._count(None)
            return "".join(pieces)

        # --- normal path: ask for n samples in ONE request ---
        # vLLM generates the n samples together and shares the prompt computation,
        # so 5 samples cost far less than 5 separate calls.
        response = self.client.chat.completions.create(n=n, **request)
        self._count(response)
        texts = [choice.message.content or "" for choice in response.choices]

        # Some servers (Ollama, llama.cpp) silently ignore `n` and return one sample.
        # Top up with extra calls. Each extra call gets its own seed: with the same seed
        # every "independent" sample would come back identical (found in the smoke test).
        while len(texts) < n:
            if seed is not None:
                request["seed"] = seed + len(texts)
            extra = self.client.chat.completions.create(n=1, **request)
            self._count(extra)
            texts.append(extra.choices[0].message.content or "")

        if n == 1:
            return texts[0]
        return texts[:n]


# ----------------------------------------------------------------------------------------
# Parsing model output
# ----------------------------------------------------------------------------------------

def parse_json(text: str) -> Optional[dict]:
    """Best-effort extraction of a JSON object from an LLM reply. None if nothing parses.

    Models wrap JSON in prose, in ```json fences, add trailing commas, or (reasoning models)
    put a <think>...</think> block first. We try the common shapes in order.
    """
    if not text:
        return None

    # Reasoning models (Qwen3, DeepSeek-R1) think out loud first; drop that part.
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

    # If there is a ```json ... ``` block, only look inside it.
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL)
    if fenced:
        text = fenced.group(1)

    # Candidate 1: the whole text. Candidate 2: from the first "{" to the last "}".
    candidates = [text.strip()]
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start:end + 1])

    for candidate in candidates:
        # Try as-is, then again with trailing commas removed ( {"a": 1,} -> {"a": 1} ).
        without_trailing_commas = re.sub(r",\s*([\]}])", r"\1", candidate)
        for attempt in (candidate, without_trailing_commas):
            try:
                value = json.loads(attempt)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                return value

    return None


def is_placeholder(text: str) -> bool:
    """True for template echoes like "...", "<query>", "" - text with no real words.
    Small models sometimes copy the example JSON from the prompt instead of filling it in."""
    text = str(text).strip()
    if text.startswith("<") and text.endswith(">"):
        return True
    words = re.findall(r"[A-Za-z]{2,}", text)
    return len(words) == 0


def parse_choice(text: str, valid: Iterable[str]) -> Optional[str]:
    """Pulls the chosen option letter (e.g. "B") out of a reply, or None if the model
    never clearly committed to one of the valid letters."""
    valid = set(valid)

    # 1) The reply is (or contains) JSON with an answer field.
    data = parse_json(text)
    if data:
        for key in ("answer_choice", "answer", "choice"):
            value = str(data.get(key, "")).strip().upper()
            # Accept "B", "(B)", "B.", "B. Heart" - but NOT "A/B/C/..." (the model echoing
            # the prompt's format hint, seen in the smoke test) and not a word like "BETA".
            match = re.fullmatch(r"\(?([A-Z])\)?(?:[.:)]?\s.*|[.:)])?", value, flags=re.DOTALL)
            if match and match.group(1) in valid:
                return match.group(1)

    # 2) Free-text fallbacks for models that ignore the JSON instruction.
    # (?!/) stops "A/B/C/..." format echoes from being read as an answer.
    patterns = [
        r'"answer_choice"\s*:\s*"?\(?([A-Z])\b(?!/)',    # broken JSON
        r"answer is:?\s*\(?([A-Z])\b(?!/)",               # "...so the answer is (B)"
        r"## Answer\s*:?\s*\(?([A-Z])\b(?!/)",            # i-MedRAG style
    ]
    for pattern in patterns:
        match = re.search(pattern, text or "", flags=re.IGNORECASE)
        if match and match.group(1).upper() in valid:
            return match.group(1).upper()

    return None


# ----------------------------------------------------------------------------------------
# Concurrency
# ----------------------------------------------------------------------------------------

def parallel_map(fn: Callable, items: List, workers: int = 8) -> Iterator:
    """Runs fn(item) for every item in a thread pool; yields (item, result) as each finishes.

    Plain threads are enough: each worker spends its time waiting for the LLM server,
    and vLLM batches all concurrent requests together on the GPU.
    """
    if workers <= 1:
        for item in items:
            yield item, fn(item)
        return

    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_to_item = {}
        for item in items:
            future = pool.submit(fn, item)
            future_to_item[future] = item

        for future in as_completed(future_to_item):
            yield future_to_item[future], future.result()
