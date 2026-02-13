"""Lightweight reranker stub for retrieval results."""

from __future__ import annotations

from dataclasses import replace
from typing import Iterable

from personal_search_layer.models import ScoredChunk


def _tokenize(text: str) -> set[str]:
    return {token for token in text.lower().split() if token}


def rerank_chunks(query: str, chunks: Iterable[ScoredChunk]) -> list[ScoredChunk]:
    """Rerank chunks using simple lexical overlap with the query.

    This is a deterministic stub meant for Week 2 wiring; replace with a model-based
    reranker later.
    """
    query_tokens = _tokenize(query)
    scored: list[ScoredChunk] = []
    for chunk in chunks:
        overlap = len(query_tokens & _tokenize(chunk.chunk_text))
        adjusted_score = chunk.score + (overlap * 0.2)
        scored.append(replace(chunk, score=adjusted_score))
    scored.sort(key=lambda item: item.score, reverse=True)
    return scored
