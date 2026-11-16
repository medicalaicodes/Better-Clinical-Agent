"""
Project settings. Each one is read from an environment variable, or from a `.env` file in the
folder you run from, or falls back to the default written below.

Example `.env` on the GPU server:

    LLM_BASE_URL=http://localhost:8000/v1
    LLM_MODEL=llama31-8b
    DATA_DIR=/path/to/big/disk/clinical-agent-data
"""

import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def read_env_file(path=".env"):
    """Reads KEY=VALUE lines from a .env file into a dictionary (blank lines and # comments skipped)."""
    values = {}
    if not os.path.exists(path):
        return values
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip().strip('"').strip("'")
    return values


class Settings:
    def __init__(self):
        file_values = read_env_file()

        def setting(name, default):
            # A real environment variable wins over the .env file, which wins over the default.
            if name in os.environ:
                return os.environ[name]
            if name in file_values:
                return file_values[name]
            return default

        # Any server that speaks the OpenAI API: vLLM, Ollama (http://localhost:11434/v1), ...
        self.llm_base_url = setting("LLM_BASE_URL", "http://localhost:8000/v1")
        self.llm_model = setting("LLM_MODEL", "llama31-8b")
        self.llm_api_key = setting("LLM_API_KEY", "EMPTY")
        self.llm_temperature = float(setting("LLM_TEMPERATURE", "0.0"))
        self.llm_max_tokens = int(setting("LLM_MAX_TOKENS", "1024"))
        self.llm_timeout = float(setting("LLM_TIMEOUT", "300"))

        # Where the corpora and indexes live (several GB, so a big disk on the server).
        self.data_dir = setting("DATA_DIR", os.path.join(REPO_ROOT, "data"))
        self.corpus_dir = os.path.join(self.data_dir, "corpus")
        self.index_dir = os.path.join(self.data_dir, "index")

        # "cuda" or "cpu"; empty means: use the GPU if there is one.
        self.device = setting("TORCH_DEVICE", None) or None


settings_cache = []


def get_settings():
    """The settings are read once and then reused."""
    if not settings_cache:
        settings_cache.append(Settings())
    return settings_cache[0]
