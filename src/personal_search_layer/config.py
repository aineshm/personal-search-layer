"""Configuration for personal_search_layer (env-overridable)."""

from __future__ import annotations

import os
from pathlib import Path


def _env_path(key: str, default: Path) -> Path:
    return Path(os.getenv(key, str(default))).expanduser()


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, default))
    except (TypeError, ValueError):
        return default


def _env_bool(key: str, default: bool) -> bool:
    raw = os.getenv(key)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = _env_path("PSL_DATA_DIR", PROJECT_ROOT / "data")
DB_PATH = DATA_DIR / "search.db"
INDEX_DIR = DATA_DIR / "indexes"
FAISS_INDEX_PATH = INDEX_DIR / "chunks.faiss"

CHUNK_SIZE = _env_int("PSL_CHUNK_SIZE", 1000)
CHUNK_OVERLAP = _env_int("PSL_CHUNK_OVERLAP", 120)
DEFAULT_TOP_K = _env_int("PSL_TOP_K", 8)

EMBEDDING_DIM = _env_int("PSL_EMBED_DIM", 384)
RRF_K = _env_int("PSL_RRF_K", 60)
MODEL_NAME = os.getenv("PSL_MODEL_NAME", "hash-embed-v1")

MAX_DOC_BYTES = _env_int("PSL_MAX_DOC_BYTES", 10_000_000)
MAX_PDF_PAGES = _env_int("PSL_MAX_PDF_PAGES", 50)
NORMALIZE_TEXT = _env_bool("PSL_NORMALIZE_TEXT", True)


def ensure_data_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
