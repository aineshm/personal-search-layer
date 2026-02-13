"""Deterministic extractive answer synthesis with claim-level citations."""

from __future__ import annotations

import re

from personal_search_layer.models import Citation, Claim, DraftAnswer, ScoredChunk
from personal_search_layer.router import PrimaryIntent

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> set[str]:
    return {token for token in _TOKEN_RE.findall(text.lower()) if token}


def _split_sentences(text: str) -> list[str]:
    sentences = [
        part.strip() for part in _SENTENCE_SPLIT_RE.split(text) if part.strip()
    ]
    return [sentence for sentence in sentences if len(sentence) >= 24]


def _claim_limit(intent: PrimaryIntent) -> int:
    if intent in {
        PrimaryIntent.SYNTHESIS,
        PrimaryIntent.COMPARE,
        PrimaryIntent.TIMELINE,
    }:
        return 5
    if intent == PrimaryIntent.TASK:
        return 4
    return 3


def _citation_for_sentence(
    claim_id: str, sentence: str, chunk: ScoredChunk
) -> Citation:
    haystack = chunk.chunk_text.lower()
    needle = sentence.lower()
    start = haystack.find(needle)
    if start < 0:
        # Fuzzy fallback: cite the opening span of the chunk.
        span = min(len(chunk.chunk_text), max(80, len(sentence)))
        return Citation(
            claim_id=claim_id,
            chunk_id=chunk.chunk_id,
            source_path=chunk.source_path,
            page=chunk.page,
            quote_span_start=0,
            quote_span_end=span,
        )
    return Citation(
        claim_id=claim_id,
        chunk_id=chunk.chunk_id,
        source_path=chunk.source_path,
        page=chunk.page,
        quote_span_start=start,
        quote_span_end=min(len(chunk.chunk_text), start + len(sentence)),
    )


def synthesize_extractive(
    query: str,
    chunks: list[ScoredChunk],
    intent: PrimaryIntent,
) -> DraftAnswer:
    """Create a deterministic extractive draft from retrieved evidence."""
    query_tokens = _tokenize(query)
    candidates: list[tuple[float, str, ScoredChunk]] = []
    for chunk in chunks:
        for sentence in _split_sentences(chunk.chunk_text):
            sentence_tokens = _tokenize(sentence)
            overlap = len(sentence_tokens & query_tokens)
            length_bonus = min(len(sentence) / 200.0, 1.0)
            score = (overlap * 2.0) + float(chunk.score) + length_bonus
            candidates.append((score, sentence, chunk))

    candidates.sort(key=lambda item: item[0], reverse=True)
    selected: list[Claim] = []
    seen_sentences: set[str] = set()
    claim_cap = _claim_limit(intent)
    for _, sentence, chunk in candidates:
        normalized = " ".join(sentence.lower().split())
        if normalized in seen_sentences:
            continue
        claim_id = f"c{len(selected) + 1}"
        citation = _citation_for_sentence(claim_id, sentence, chunk)
        selected.append(Claim(claim_id=claim_id, text=sentence, citations=[citation]))
        seen_sentences.add(normalized)
        if len(selected) >= claim_cap:
            break

    if not selected and chunks:
        claim_id = "c1"
        fallback = chunks[0].chunk_text.strip()[:200]
        selected = [
            Claim(
                claim_id=claim_id,
                text=fallback,
                citations=[_citation_for_sentence(claim_id, fallback, chunks[0])],
            )
        ]

    answer_text = "\n".join(f"- {claim.text}" for claim in selected)
    return DraftAnswer(answer_text=answer_text, claims=selected)
