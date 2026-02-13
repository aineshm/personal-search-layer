"""Deterministic follow-up query proposal for bounded multi-hop."""

from __future__ import annotations

import re

from personal_search_layer.models import DraftAnswer

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return [token for token in _TOKEN_RE.findall(text.lower()) if token]


def propose_followup_query(
    query: str,
    draft: DraftAnswer | None,
    missing_claims: list[str],
) -> str | None:
    """Build a single deterministic follow-up query from missing evidence signals."""
    if not missing_claims and not draft:
        return None

    seed_text = " ".join(missing_claims)
    if not seed_text and draft:
        seed_text = " ".join(claim.text for claim in draft.claims[:2])
    if not seed_text.strip():
        return None

    original = set(_tokenize(query))
    additions: list[str] = []
    for token in _tokenize(seed_text):
        if len(token) < 4:
            continue
        if token in original:
            continue
        if token in additions:
            continue
        additions.append(token)
        if len(additions) >= 6:
            break

    if not additions:
        return None
    return f"{query} {' '.join(additions)}"
