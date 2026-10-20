from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

DEFAULT_EMBEDDING_MODELS = {
    "fastembed": "BAAI/bge-small-en-v1.5",
    "openai": "text-embedding-3-small",
}


@dataclass(frozen=True)
class Settings:
    root: Path
    embedding_provider: str
    embedding_model: str
    llm_base_url: str | None
    llm_model: str
    ollama_model: str
    ollama_base_url: str

    @property
    def registry_path(self) -> Path:
        return self.root / "data" / "universities.json"

    @property
    def artifacts_dir(self) -> Path:
        return self.root / "artifacts"

    @property
    def raw_pages_dir(self) -> Path:
        return self.root / "raw_pages"


def get_settings() -> Settings:
    """Load environment variables from the project-local .env file."""
    root = Path(__file__).resolve().parents[2]
    load_dotenv(root / ".env")
    provider = os.getenv("EMBEDDING_PROVIDER", "fastembed").strip().lower()
    if provider not in DEFAULT_EMBEDDING_MODELS:
        raise RuntimeError(f"EMBEDDING_PROVIDER must be one of {sorted(DEFAULT_EMBEDDING_MODELS)}, got {provider!r}.")
    return Settings(
        root=root,
        embedding_provider=provider,
        embedding_model=os.getenv("EMBEDDING_MODEL") or DEFAULT_EMBEDDING_MODELS[provider],
        # Any OpenAI-compatible chat endpoint (Groq, OpenRouter, Ollama's /v1, ...).
        # Empty means the OpenAI API itself.
        llm_base_url=os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1") or None,
        llm_model=os.getenv("LLM_MODEL", "openai/gpt-oss-120b"),
        ollama_model=os.getenv("OLLAMA_MODEL", "mistral"),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    )


def require_openai_key() -> str:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY is missing. It is only needed when EMBEDDING_PROVIDER=openai or LLM_BASE_URL is empty."
        )
    return key


def require_llm_key() -> str:
    key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("LLM_API_KEY is missing. Add your Groq (or other provider) key to .env.")
    return key
