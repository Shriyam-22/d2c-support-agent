from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:
        pass


@dataclass
class Settings:
    llm_api_key: str
    llm_model: str
    llm_base_url: str | None
    retriever: str  # "bm25" | "embedding"
    embedding_model: str
    data_dir: Path
    log_dir: Path | None
    max_steps: int

    @classmethod
    def from_env(cls) -> "Settings":
        _load_dotenv()
        log_dir = os.getenv("LOG_DIR", "logs")
        return cls(
            llm_api_key=os.getenv("LLM_API_KEY", ""),
            llm_model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
            llm_base_url=os.getenv("LLM_BASE_URL") or None,
            retriever=os.getenv("RETRIEVER", "bm25").lower(),
            embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
            data_dir=Path(os.getenv("DATA_DIR", ROOT / "data")),
            log_dir=(ROOT / log_dir) if log_dir else None,
            max_steps=int(os.getenv("MAX_STEPS", "5")),
        )
