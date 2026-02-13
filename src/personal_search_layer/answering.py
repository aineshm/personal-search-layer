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
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "what",
    "when",
    "where",
    "which",
    "with",
}


@dataclass(frozen=True)
class _Candidate:
    sentence: str
    chunk: ScoredChunk
    overlap_score: float
    supportability_score: float
    citation_span_quality: float
    source_count: int
    stage_score: float
    signature: str
    sentence_tokens: set[str]
    semantic_tokens: set[str]


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


def _normalize_token(token: str) -> str:
    if len(token) <= 4:
        return token
    if token.endswith("ies") and len(token) > 5:
        return token[:-3] + "y"
    if token.endswith("ing") and len(token) > 6:
        return token[:-3]
    if token.endswith("ed") and len(token) > 5:
        return token[:-2]
    if token.endswith("s") and len(token) > 4:
        return token[:-1]
    return token[:6] if len(token) > 6 else token


def _semantic_tokens(sentence: str) -> set[str]:
    return {
        _normalize_token(token)
        for token in _TOKEN_RE.findall(sentence.lower())
        if token not in _STOPWORDS and len(token) >= 3
    }


def _claim_signature(sentence: str) -> str:
    tokens = _semantic_tokens(sentence)
    if not tokens:
        return ""
    deduped = sorted({token[:5] if len(token) > 5 else token for token in tokens})
    return " ".join(deduped[:12])


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
        span_text = chunk.chunk_text[:span].lower()
        sentence_tokens = _tokenize(sentence)
        overlap = len(sentence_tokens & _tokenize(span_text)) / max(
            1, len(sentence_tokens)
        )
        quality = min(1.0, span / max(1, len(chunk.chunk_text))) * 0.4 + overlap * 0.4
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
    span_text = chunk.chunk_text[start:end].lower()
    sentence_tokens = _tokenize(sentence)
    overlap = len(sentence_tokens & _tokenize(span_text)) / max(1, len(sentence_tokens))
    quality = min(1.0, span_len / max(1, len(sentence))) * 0.7 + overlap * 0.3
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
        signature=_claim_signature(sentence),
        sentence_tokens=sentence_tokens,
        semantic_tokens=_semantic_tokens(sentence),
    )


def _group_candidates(candidates: list[_Candidate]) -> list[list[_Candidate]]:
    grouped: list[list[_Candidate]] = []
    for candidate in candidates:
        attached = False
        for group in grouped:
            rep = group[0]
            if candidate.signature and candidate.signature == rep.signature:
                group.append(candidate)
                attached = True
                break
            overlap = candidate.semantic_tokens & rep.semantic_tokens
            union = candidate.semantic_tokens | rep.semantic_tokens
            if not union:
                continue
            jaccard = len(overlap) / len(union)
            containment = len(overlap) / max(
                1, min(len(candidate.semantic_tokens), len(rep.semantic_tokens))
            )
            if jaccard >= 0.6 or containment >= 0.7:
                group.append(candidate)
                attached = True
                break
        if not attached:
            grouped.append([candidate])
    return grouped


def _representative_candidate(group: list[_Candidate]) -> _Candidate:
    if len(group) == 1:
        return group[0]

    def score(candidate: _Candidate) -> tuple[int, float, float, int]:
        source_best: dict[str, float] = {}
        for peer in group:
            _, quality = _citation_for_sentence("tmp", candidate.sentence, peer.chunk)
            source_best[peer.chunk.source_path] = max(
                source_best.get(peer.chunk.source_path, 0.0), quality
            )
        supported_sources = sum(
            1
            for quality in source_best.values()
            if quality >= ANSWER_MIN_CITATION_SPAN_QUALITY
        )
        avg_quality = sum(source_best.values()) / max(1, len(source_best))
        # Favor concise representative text when quality/support ties.
        return (
            supported_sources,
            avg_quality,
            candidate.stage_score,
            -len(candidate.sentence),
        )

    return max(group, key=score)


def synthesize_extractive(
    query: str,
    chunks: list[ScoredChunk],
    intent: PrimaryIntent,
) -> DraftAnswer:
    """Create a deterministic extractive draft from retrieved evidence."""
    query_tokens = _tokenize(query)
    topical_floor = ANSWER_MIN_TOPIC_OVERLAP
    if intent in {PrimaryIntent.FACT, PrimaryIntent.OTHER, PrimaryIntent.TASK}:
        topical_floor = max(topical_floor, 2)

    # Stage 1: candidate generation
    candidates: list[_Candidate] = []
    for chunk in chunks:
        for sentence in _split_sentences(chunk.chunk_text):
            candidates.append(_candidate_stage(sentence, chunk, query_tokens))

    # Stage 2: topical alignment filter
    topical = [
        cand
        for cand in candidates
        if len(cand.sentence_tokens & query_tokens) >= topical_floor
    ]

    # Stage 3: supportability filter
    supportable = [
        cand
        for cand in topical
        if cand.supportability_score >= ANSWER_MIN_SUPPORTABILITY
    ]

    grouped = _group_candidates(supportable)
    grouped.sort(
        key=lambda group: (
            len({cand.chunk.source_path for cand in group}),
            max(cand.stage_score for cand in group),
            sum(cand.stage_score for cand in group) / max(1, len(group)),
        ),
        reverse=True,
    )

    # Stage 4: final claims with dedupe and citation quality check
    selected: list[Claim] = []
    seen_signatures: set[str] = set()
    claim_cap = _claim_limit(intent)
    prefer_multi_source = intent in {
        PrimaryIntent.SYNTHESIS,
        PrimaryIntent.COMPARE,
        PrimaryIntent.TIMELINE,
    }

    ordered_groups = grouped
    if prefer_multi_source:
        multi = [
            group
            for group in grouped
            if len({cand.chunk.source_path for cand in group}) >= 2
        ]
        single = [
            group
            for group in grouped
            if len({cand.chunk.source_path for cand in group}) < 2
        ]
        ordered_groups = multi + single

    for group in ordered_groups:
        best = _representative_candidate(group)
        if not best.signature or best.signature in seen_signatures:
            continue

        claim_id = f"c{len(selected) + 1}"
        unique_sources: set[str] = set()
        citations: list[Citation] = []
        citation_qualities: list[float] = []
        for cand in sorted(group, key=lambda item: item.stage_score, reverse=True):
            if cand.chunk.source_path in unique_sources:
                continue
            citation, span_quality = _citation_for_sentence(
                claim_id, best.sentence, cand.chunk
            )
            if span_quality < ANSWER_MIN_CITATION_SPAN_QUALITY:
                continue
            citations.append(citation)
            citation_qualities.append(span_quality)
            unique_sources.add(cand.chunk.source_path)
            if len(citations) >= 2:
                break
        if not citations:
            continue

        selected.append(
            Claim(
                claim_id=claim_id,
                text=best.sentence,
                citations=citations,
                overlap_score=max(cand.overlap_score for cand in group),
                citation_span_quality=max(citation_qualities),
                source_count=len(unique_sources),
                supportability_score=max(cand.supportability_score for cand in group),
            )
        )
        seen_signatures.add(best.signature)
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
