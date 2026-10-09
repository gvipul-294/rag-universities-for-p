from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    root: Path
    embedding_model: str
    chat_model: str

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
    return Settings(
        root=root,
        embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        chat_model=os.getenv("OPENAI_CHAT_MODEL", "gpt-5-mini"),
    )


def require_api_key() -> str:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY is missing. Copy .env.example to .env and add your key."
        )
    return key

