"""
Clinical AI Agent: Configuration Settings
Provides centralized configuration for LLM backends, execution parameters, and caching.
"""

import os
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="allow")

    # LLM reasoning backend (e.g., 'ollama/medgemma', 'gemini/gemini-1.5-flash', 'gpt-4o-mini', 'fallback')
    llm_model: str = Field(default="ollama/medgemma", alias="LLM_MODEL")
    llm_temperature: float = Field(default=0.1, alias="LLM_TEMPERATURE")
    llm_max_tokens: int = Field(default=1500, alias="LLM_MAX_TOKENS")
    llm_timeout: float = Field(default=60.0, alias="LLM_TIMEOUT")
    
    # Execution & Active Inquiry
    max_inquiry_turns: int = Field(default=4, alias="MAX_INQUIRY_TURNS")
    device: str = Field(default="cpu", alias="TORCH_DEVICE")
    
    # Caching
    cache_dir: str = Field(default=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "cache"))
    statpearls_cache_dir: str = Field(default=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "cache", "statpearls"))


_SETTINGS_INSTANCE: Optional[Settings] = None


def get_settings() -> Settings:
    global _SETTINGS_INSTANCE
    if _SETTINGS_INSTANCE is None:
        _SETTINGS_INSTANCE = Settings()
        os.makedirs(_SETTINGS_INSTANCE.cache_dir, exist_ok=True)
        os.makedirs(_SETTINGS_INSTANCE.statpearls_cache_dir, exist_ok=True)
    return _SETTINGS_INSTANCE
