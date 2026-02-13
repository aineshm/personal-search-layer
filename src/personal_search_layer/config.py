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


def _env_suffix_set(key: str, default: set[str]) -> set[str]:
    raw = os.getenv(key)
    if raw is None:
        return set(default)
    parts = [part.strip().lower() for part in raw.split(",") if part.strip()]
    suffixes: set[str] = set()
    for part in parts:
        if not part:
            continue
        suffixes.add(part if part.startswith(".") else f".{part}")
    return suffixes


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = _env_path("PSL_DATA_DIR", PROJECT_ROOT / "data")
DB_PATH = DATA_DIR / "search.db"
INDEX_DIR = DATA_DIR / "indexes"
FAISS_INDEX_PATH = INDEX_DIR / "chunks.faiss"

CHUNK_SIZE = _env_int("PSL_CHUNK_SIZE", 1500)
CHUNK_OVERLAP = _env_int("PSL_CHUNK_OVERLAP", 150)
DEFAULT_TOP_K = _env_int("PSL_TOP_K", 8)

EMBEDDING_BACKEND = os.getenv("PSL_EMBEDDING_BACKEND", "sentence-transformers")
EMBEDDING_DIM = _env_int("PSL_EMBED_DIM", 384)
EMBEDDING_BATCH_SIZE = _env_int("PSL_EMBED_BATCH_SIZE", 64)
RRF_K = _env_int("PSL_RRF_K", 60)
MODEL_NAME = os.getenv("PSL_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
MODEL_REVISION = os.getenv("PSL_MODEL_REVISION", "").strip() or None

MAX_DOC_BYTES = _env_int("PSL_MAX_DOC_BYTES", 30_000_000)
MAX_PDF_PAGES = _env_int("PSL_MAX_PDF_PAGES", 200)
NORMALIZE_TEXT = _env_bool("PSL_NORMALIZE_TEXT", True)
BLOCKED_SUFFIXES = _env_suffix_set(
    "PSL_BLOCKED_SUFFIXES",
    {".json", ".csv", ".tsv", ".png", ".zip"},
)

ANSWER_MIN_TOPIC_OVERLAP = _env_int("PSL_ANSWER_MIN_TOPIC_OVERLAP", 1)
ANSWER_MIN_SUPPORTABILITY = _env_int("PSL_ANSWER_MIN_SUPPORTABILITY", 35) / 100.0
ANSWER_MIN_CITATION_SPAN_QUALITY = (
    _env_int("PSL_ANSWER_MIN_CITATION_SPAN_QUALITY", 40) / 100.0
)

VERIFIER_QUERY_ALIGNMENT_MIN = _env_int("PSL_VERIFIER_QUERY_ALIGNMENT_MIN", 30) / 100.0
VERIFIER_CRITICAL_COVERAGE_MIN = (
    _env_int("PSL_VERIFIER_CRITICAL_COVERAGE_MIN", 50) / 100.0
)
VERIFIER_CLAIM_SUPPORT_MIN = _env_int("PSL_VERIFIER_CLAIM_SUPPORT_MIN", 60) / 100.0
VERIFIER_CITATION_SPAN_QUALITY_MIN = (
    _env_int("PSL_VERIFIER_CITATION_SPAN_QUALITY_MIN", 45) / 100.0
)
VERIFIER_AGGREGATE_MIN = _env_int("PSL_VERIFIER_AGGREGATE_MIN", 55) / 100.0


def ensure_data_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
