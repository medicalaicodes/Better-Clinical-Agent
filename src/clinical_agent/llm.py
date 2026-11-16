"""
One small client for every call to the language model (the agent and the benchmarks).

It talks to any server that speaks the OpenAI chat API:
    - vLLM on the GPU server   (what the benchmarks use)
    - Ollama on a laptop       (set LLM_BASE_URL=http://localhost:11434/v1)

It also counts calls and tokens per thread. The benchmark answers every question inside its
own worker thread, so after a question is done, usage() says exactly what that question cost.
"""

import json
import re
import threading

from openai import OpenAI

from .config import get_settings


class LLM:
    def __init__(self, model=None, base_url=None, api_key=None, temperature=None, max_tokens=None):
        settings = get_settings()

        # Anything not given here comes from the .env file.
        self.model = model or settings.llm_model
        self.max_tokens = max_tokens or settings.llm_max_tokens
        if temperature is None:
            self.temperature = settings.llm_temperature
        else:
            self.temperature = temperature

        self.client = OpenAI(
            base_url=base_url or settings.llm_base_url,
            api_key=api_key or settings.llm_api_key,    # vLLM ignores it, but the library wants one
            timeout=settings.llm_timeout,
            max_retries=3,                              # the library retries timeouts for us
        )

        # threading.local() gives every thread its own private counters.
        self.counters = threading.local()

    # ------------------------------------------------------------------------------------
    # Cost counting (separately for every thread)
    # ------------------------------------------------------------------------------------
    def reset_usage(self):
        self.counters.calls = 0
        self.counters.prompt_tokens = 0
        self.counters.completion_tokens = 0

    def usage(self):
        if not hasattr(self.counters, "calls"):
            self.reset_usage()
        return {
            "calls": self.counters.calls,
            "prompt_tokens": self.counters.prompt_tokens,
            "completion_tokens": self.counters.completion_tokens,
        }

    def count(self, response):
        if not hasattr(self.counters, "calls"):
            self.reset_usage()
        self.counters.calls += 1
        # A streamed reply has no token counts, so we only add them when they're there.
        if response is not None and response.usage is not None:
            self.counters.prompt_tokens += response.usage.prompt_tokens or 0
            self.counters.completion_tokens += response.usage.completion_tokens or 0

    # ------------------------------------------------------------------------------------
    # The one method everything uses
    # ------------------------------------------------------------------------------------
    def chat(self, prompt, system=None, temperature=None, max_tokens=None, n=1, seed=None,
             on_token=None, json_mode=False):
        """Sends one prompt and returns the reply text (or a list of n replies if n > 1).

        n          several independent answers in ONE request (used by self-consistency)
        seed       makes sampled answers repeatable
        on_token   prints the reply word by word while it's written (the interactive demo)
        json_mode  vLLM may then only write valid JSON. Without it, about half of the agent's
                   long JSON replies came back broken in the first test run.
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
            request["seed"] = seed
        if json_mode:
            request["response_format"] = {"type": "json_object"}

        # Streaming: hand each new piece of text to on_token as soon as it arrives.
        if on_token is not None:
            pieces = []
            for chunk in self.client.chat.completions.create(stream=True, **request):
                if chunk.choices and chunk.choices[0].delta.content:
                    piece = chunk.choices[0].delta.content
                    pieces.append(piece)
                    on_token(piece)
            self.count(None)
            return "".join(pieces)

        # Normal: ask for all n answers in one request. vLLM reads the prompt once and
        # writes the n answers side by side, so 5 answers cost far less than 5 calls.
        response = self.client.chat.completions.create(n=n, **request)
        self.count(response)
        texts = []
        for choice in response.choices:
            texts.append(choice.message.content or "")

        # Some servers (Ollama) ignore n and send back just one answer; ask again for the rest.
        # Each extra request needs its own seed - with the same seed they'd all be identical.
        while len(texts) < n:
            if seed is not None:
                request["seed"] = seed + len(texts)
            extra = self.client.chat.completions.create(n=1, **request)
            self.count(extra)
            texts.append(extra.choices[0].message.content or "")

        if n == 1:
            return texts[0]
        return texts


# ----------------------------------------------------------------------------------------
# Reading what the model wrote
# ----------------------------------------------------------------------------------------

def parse_json(text):
    """Finds the JSON object in a model's reply. Returns a dict, or None if there isn't one.

    Models wrap JSON in sentences or in ```json fences, leave a comma before a closing bracket,
    and reasoning models write <think>...</think> first. We try the usual shapes one by one.
    """
    if not text:
        return None

    # Drop a <think>...</think> block if there is one.
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

    # If there's a ```json ... ``` block, only look inside it.
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL)
    if fenced:
        text = fenced.group(1)

    # Try the whole text, then just the part from the first "{" to the last "}".
    candidates = [text.strip()]
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start:end + 1])

    for candidate in candidates:
        # {"a": 1,} is not valid JSON; also try with such trailing commas removed.
        without_trailing_commas = re.sub(r",\s*([\]}])", r"\1", candidate)
        for attempt in [candidate, without_trailing_commas]:
            try:
                value = json.loads(attempt)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                return value
    return None


def is_placeholder(text):
    """True for text that is really just the prompt's example copied back, like "<query>" or "...".
    Small models sometimes do that instead of filling the example in."""
    text = str(text).strip()
    if text.startswith("<") and text.endswith(">"):
        return True
    real_words = re.findall(r"[A-Za-z]{2,}", text)
    return len(real_words) == 0


def parse_choice(text, valid):
    """The option letter the model chose (like "B"), or None if it never clearly picked one."""
    valid = list(valid)

    # 1) The reply is JSON with an answer field. We accept "B", "(B)", "B." and "B. Heart",
    #    but not "A/B/C/..." (the model copying the format hint) and not a word like "BETA".
    data = parse_json(text)
    if data:
        for key in ["answer_choice", "answer", "choice"]:
            value = str(data.get(key, "")).strip().upper()
            match = re.fullmatch(r"\(?([A-Z])\)?(?:[.:)]?\s.*|[.:)])?", value, flags=re.DOTALL)
            if match and match.group(1) in valid:
                return match.group(1)

    # 2) No usable JSON: look for the usual ways of stating an answer in plain text.
    #    (?!/) again stops "A/B/C/..." from counting as an answer.
    patterns = [
        r'"answer_choice"\s*:\s*"?\(?([A-Z])\b(?!/)',     # broken JSON
        r"answer is:?\s*\(?([A-Z])\b(?!/)",                # "...so the answer is (B)"
        r"## Answer\s*:?\s*\(?([A-Z])\b(?!/)",             # i-MedRAG's style
    ]
    for pattern in patterns:
        match = re.search(pattern, text or "", flags=re.IGNORECASE)
        if match and match.group(1).upper() in valid:
            return match.group(1).upper()
    return None
