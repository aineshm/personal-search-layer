"""Deterministic extractive answer synthesis with claim-level citations."""

from __future__ import annotations

import re
from dataclasses import dataclass

from personal_search_layer.config import (
    ANSWER_MIN_CITATION_SPAN_QUALITY,
    ANSWER_MIN_SUPPORTABILITY,
    ANSWER_MIN_TOPIC_OVERLAP,
)
from personal_search_layer.models import Citation, Claim, DraftAnswer, ScoredChunk
from personal_search_layer.router import PrimaryIntent

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_TOKEN_RE = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class _Candidate:
    sentence: str
    chunk: ScoredChunk
    overlap_score: float
    supportability_score: float
    citation_span_quality: float
    source_count: int
    stage_score: float


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


def _supportability(sentence_tokens: set[str], chunk_tokens: set[str]) -> float:
    if not sentence_tokens:
        return 0.0
    return len(sentence_tokens & chunk_tokens) / len(sentence_tokens)


def _citation_for_sentence(
    claim_id: str,
    sentence: str,
    chunk: ScoredChunk,
) -> tuple[Citation, float]:
    haystack = chunk.chunk_text.lower()
    needle = sentence.lower()
    start = haystack.find(needle)
    if start < 0:
        span = min(len(chunk.chunk_text), max(80, len(sentence)))
        quality = min(1.0, span / max(1, len(chunk.chunk_text))) * 0.6
        return (
            Citation(
                claim_id=claim_id,
                chunk_id=chunk.chunk_id,
                source_path=chunk.source_path,
                page=chunk.page,
                quote_span_start=0,
                quote_span_end=span,
            ),
            quality,
        )

    end = min(len(chunk.chunk_text), start + len(sentence))
    span_len = max(1, end - start)
    quality = min(1.0, span_len / max(1, len(sentence)))
    return (
        Citation(
            claim_id=claim_id,
            chunk_id=chunk.chunk_id,
            source_path=chunk.source_path,
            page=chunk.page,
            quote_span_start=start,
            quote_span_end=end,
        ),
        quality,
    )


def _candidate_stage(
    sentence: str,
    chunk: ScoredChunk,
    query_tokens: set[str],
) -> _Candidate:
    sentence_tokens = _tokenize(sentence)
    chunk_tokens = _tokenize(chunk.chunk_text)
    overlap_count = len(sentence_tokens & query_tokens)
    overlap_score = overlap_count / max(1, len(query_tokens))
    supportability_score = _supportability(sentence_tokens, chunk_tokens)
    # Temporary span quality using sentence/chunk ratio; final citation uses exact span.
    citation_span_quality = min(1.0, len(sentence) / max(1, len(chunk.chunk_text)))
    stage_score = (
        float(chunk.score)
        + overlap_score * 1.2
        + supportability_score * 1.0
        + citation_span_quality * 0.6
    )
    return _Candidate(
        sentence=sentence,
        chunk=chunk,
        overlap_score=overlap_score,
        supportability_score=supportability_score,
        citation_span_quality=citation_span_quality,
        source_count=1,
        stage_score=stage_score,
    )


def synthesize_extractive(
    query: str,
    chunks: list[ScoredChunk],
    intent: PrimaryIntent,
) -> DraftAnswer:
    """Create a deterministic extractive draft from retrieved evidence."""
    query_tokens = _tokenize(query)

    # Stage 1: candidate generation
    candidates: list[_Candidate] = []
    for chunk in chunks:
        for sentence in _split_sentences(chunk.chunk_text):
            candidates.append(_candidate_stage(sentence, chunk, query_tokens))

    # Stage 2: topical alignment filter
    topical = [
        cand
        for cand in candidates
        if len(_tokenize(cand.sentence) & query_tokens) >= ANSWER_MIN_TOPIC_OVERLAP
    ]

    # Stage 3: supportability filter
    supportable = [
        cand
        for cand in topical
        if cand.supportability_score >= ANSWER_MIN_SUPPORTABILITY
    ]

    supportable.sort(key=lambda item: item.stage_score, reverse=True)

    # Stage 4: final claims with dedupe and citation quality check
    selected: list[Claim] = []
    seen_sentences: set[str] = set()
    claim_cap = _claim_limit(intent)
    for cand in supportable:
        normalized = " ".join(cand.sentence.lower().split())
        if normalized in seen_sentences:
            continue

        claim_id = f"c{len(selected) + 1}"
        citation, span_quality = _citation_for_sentence(
            claim_id, cand.sentence, cand.chunk
        )
        if span_quality < ANSWER_MIN_CITATION_SPAN_QUALITY:
            continue

        selected.append(
            Claim(
                claim_id=claim_id,
                text=cand.sentence,
                citations=[citation],
                overlap_score=cand.overlap_score,
                citation_span_quality=span_quality,
                source_count=cand.source_count,
                supportability_score=cand.supportability_score,
            )
        )
        seen_sentences.add(normalized)
        if len(selected) >= claim_cap:
            break

    if not selected and chunks:
        # Fallback remains deterministic but low confidence by construction.
        claim_id = "c1"
        fallback = chunks[0].chunk_text.strip()[:200]
        citation, span_quality = _citation_for_sentence(claim_id, fallback, chunks[0])
        selected = [
            Claim(
                claim_id=claim_id,
                text=fallback,
                citations=[citation],
                overlap_score=0.0,
                citation_span_quality=span_quality,
                source_count=1,
                supportability_score=0.0,
            )
        ]

    answer_text = "\n".join(f"- {claim.text}" for claim in selected)
    return DraftAnswer(answer_text=answer_text, claims=selected)
