"""
Project settings, read from environment variables or a local `.env` file.

Example `.env` for the benchmark server:

    LLM_BASE_URL=http://localhost:8000/v1
    LLM_MODEL=llama31-8b
    DATA_DIR=/path/to/big/disk/clinical-agent-data
"""

import os
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Any OpenAI-compatible server: vLLM, Ollama (http://localhost:11434/v1), llama.cpp, OpenAI.
    llm_base_url: str = Field(default="http://localhost:8000/v1", alias="LLM_BASE_URL")
    llm_model: str = Field(default="llama31-8b", alias="LLM_MODEL")
    llm_api_key: str = Field(default="EMPTY", alias="LLM_API_KEY")
    llm_temperature: float = Field(default=0.0, alias="LLM_TEMPERATURE")
    llm_max_tokens: int = Field(default=1024, alias="LLM_MAX_TOKENS")
    llm_timeout: float = Field(default=300.0, alias="LLM_TIMEOUT")

    # Where corpora and indexes live (several GB, so point this at a big disk on the server).
    data_dir: str = Field(default=os.path.join(REPO_ROOT, "data"), alias="DATA_DIR")
    device: Optional[str] = Field(default=None, alias="TORCH_DEVICE")  # None = cuda if available

    # Agent loop
    max_inquiry_turns: int = Field(default=4, alias="MAX_INQUIRY_TURNS")

    @property
    def corpus_dir(self) -> str:
        return os.path.join(self.data_dir, "corpus")

    @property
    def index_dir(self) -> str:
        return os.path.join(self.data_dir, "index")


_SETTINGS: Optional[Settings] = None


def get_settings() -> Settings:
    global _SETTINGS
    if _SETTINGS is None:
        _SETTINGS = Settings()
    return _SETTINGS
