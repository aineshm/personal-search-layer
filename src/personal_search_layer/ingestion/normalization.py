"""Text normalization helpers for ingestion.

Normalization is applied before chunking/indexing to improve lexical matching,
dedupe stability, and overall retrieval robustness.
"""

from __future__ import annotations

import re
import unicodedata


_WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Normalize text for indexing.

    Steps:
    - NFKC normalization to reduce unicode variants.
    - Lowercase for lexical match consistency.
    - Collapse whitespace to single spaces.
    """
    normalized = unicodedata.normalize("NFKC", text)
    normalized = normalized.lower()
    normalized = _WHITESPACE_RE.sub(" ", normalized).strip()
    return normalized
