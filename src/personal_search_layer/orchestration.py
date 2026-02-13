"""Core query orchestration for search and answer modes."""

from __future__ import annotations

import time
from dataclasses import replace
from typing import Literal

from personal_search_layer.answering import synthesize_extractive
from personal_search_layer.models import DraftAnswer, OrchestrationResult, ScoredChunk
from personal_search_layer.multihop import propose_followup_query
from personal_search_layer.rerank import rerank_chunks
from personal_search_layer.retrieval import fuse_hybrid, search_lexical, search_vector
from personal_search_layer.router import PrimaryIntent, PipelineSettings, route_query
from personal_search_layer.verification import repair_answer, verify_answer

MAX_HOPS = 1
MAX_REPAIRS = 1


def _enforce_pipeline_bounds(settings: PipelineSettings) -> PipelineSettings:
    allow_multihop = max(0, min(settings.allow_multihop, MAX_HOPS))
    max_repair_passes = max(0, min(settings.max_repair_passes, MAX_REPAIRS))
    if allow_multihop == 0:
        max_repair_passes = 0
    if (
        allow_multihop == settings.allow_multihop
        and max_repair_passes == settings.max_repair_passes
    ):
        return settings
    return replace(
        settings,
        allow_multihop=allow_multihop,
        max_repair_passes=max_repair_passes,
    )


def _merge_chunks(
    primary: list[ScoredChunk], secondary: list[ScoredChunk]
) -> list[ScoredChunk]:
    by_id: dict[str, ScoredChunk] = {chunk.chunk_id: chunk for chunk in primary}
    for chunk in secondary:
        existing = by_id.get(chunk.chunk_id)
        if existing is None or chunk.score > existing.score:
            by_id[chunk.chunk_id] = chunk
    return sorted(by_id.values(), key=lambda item: item.score, reverse=True)


def _run_retrieval(query: str, *, top_k: int, skip_vector: bool, lexical_weight: float):
    lexical = search_lexical(query, k=top_k)
    vector = search_vector(query, k=top_k) if not skip_vector else None
    hybrid = (
        fuse_hybrid(lexical, vector, k=top_k, lexical_weight=lexical_weight)
        if vector
        else lexical
    )
    return lexical, vector, hybrid


def _missing_claims(draft: DraftAnswer, verification) -> list[str]:
    bad_claims = {
        issue.claim_id
        for issue in verification.issues
        if issue.claim_id and issue.type in {"unsupported_claim", "missing_citation"}
    }
    missing: list[str] = []
    for claim in draft.claims:
        if claim.claim_id in bad_claims:
            missing.append(claim.text)
    return missing


def run_query(
    query: str,
    mode: Literal["search", "answer"],
    top_k: int | None = None,
    skip_vector: bool | None = None,
) -> OrchestrationResult:
    start = time.perf_counter()
    decision = route_query(query)
    settings = _enforce_pipeline_bounds(decision.recommended_pipeline_settings)

    effective_top_k = top_k if top_k is not None else settings.k
    intent = decision.primary_intent
    effective_skip_vector = (
        skip_vector if skip_vector is not None else intent == PrimaryIntent.LOOKUP
    )
    use_rerank = settings.use_rerank and intent in {
        PrimaryIntent.SYNTHESIS,
        PrimaryIntent.TASK,
        PrimaryIntent.COMPARE,
        PrimaryIntent.TIMELINE,
    }

    searched_queries = [query]
    hop_count = 0
    repair_count = 0
    repair_outcome = "none"
    verifier_timing_ms: dict[str, float] = {}

    lexical, vector, hybrid = _run_retrieval(
        query,
        top_k=effective_top_k,
        skip_vector=effective_skip_vector,
        lexical_weight=settings.lexical_weight,
    )
    chunks = hybrid.chunks
    if use_rerank:
        chunks = rerank_chunks(query, chunks)

    draft_answer = None
    verification = None

    if mode == "answer":
        draft_answer = synthesize_extractive(query, chunks, intent)
        draft_answer.searched_queries = list(searched_queries)
        verify_start = time.perf_counter()
        verification = verify_answer(
            query, draft_answer, chunks, settings.verifier_mode, intent=intent
        )
        verifier_timing_ms["initial_verify"] = (
            time.perf_counter() - verify_start
        ) * 1000
        verification.searched_queries = list(searched_queries)

        if (
            verification.abstain
            and settings.allow_multihop == 1
            and hop_count < MAX_HOPS
        ):
            followup = propose_followup_query(
                query,
                draft_answer,
                _missing_claims(draft_answer, verification),
            )
            if followup and followup not in searched_queries:
                searched_queries.append(followup)
                hop_count += 1
                _, _, hop_hybrid = _run_retrieval(
                    followup,
                    top_k=effective_top_k,
                    skip_vector=effective_skip_vector,
                    lexical_weight=settings.lexical_weight,
                )
                chunks = _merge_chunks(chunks, hop_hybrid.chunks)
                if use_rerank:
                    chunks = rerank_chunks(query, chunks)
                draft_answer = synthesize_extractive(query, chunks, intent)
                draft_answer.searched_queries = list(searched_queries)
                verify_start = time.perf_counter()
                verification = verify_answer(
                    query, draft_answer, chunks, settings.verifier_mode, intent=intent
                )
                verifier_timing_ms["post_hop_verify"] = (
                    time.perf_counter() - verify_start
                ) * 1000
                verification.searched_queries = list(searched_queries)

        if (
            verification
            and verification.abstain
            and settings.max_repair_passes > 0
            and repair_count < MAX_REPAIRS
        ):
            if verification.verdict_code in {
                "query_mismatch",
                "conflict_detected",
                "insufficient_evidence",
            }:
                repair_outcome = "skipped_ineligible"
            else:
                repair_outcome = "noop"
            repaired = None
            if repair_outcome == "noop":
                repaired = repair_answer(
                    query,
                    draft_answer,
                    chunks,
                    settings.verifier_mode,
                    intent=intent,
                )
                repair_count += 1
                if repaired is not None:
                    repaired.searched_queries = list(searched_queries)
                    draft_answer = repaired
                    verify_start = time.perf_counter()
                    verification = verify_answer(
                        query,
                        draft_answer,
                        chunks,
                        settings.verifier_mode,
                        intent=intent,
                    )
                    verifier_timing_ms["post_repair_verify"] = (
                        time.perf_counter() - verify_start
                    ) * 1000
                    verification.searched_queries = list(searched_queries)
                    repair_outcome = (
                        "successful" if not verification.abstain else "harmful"
                    )
                else:
                    repair_outcome = "unsuccessful"

            if verification and verification.abstain:
                verification.searched_queries = list(searched_queries)

    elapsed_ms = (time.perf_counter() - start) * 1000
    tool_trace = {
        "router": {
            "primary_intent": decision.primary_intent.value,
            "signals": decision.signals,
            "settings": {
                "k": settings.k,
                "lexical_weight": settings.lexical_weight,
                "allow_multihop": settings.allow_multihop,
                "use_rerank": settings.use_rerank,
                "generate_answer": settings.generate_answer,
                "verifier_mode": settings.verifier_mode.value,
                "max_repair_passes": settings.max_repair_passes,
            },
        },
        "retrieval": {
            "top_k": effective_top_k,
            "skip_vector": effective_skip_vector,
            "lexical_latency_ms": lexical.latency_ms,
            "vector_latency_ms": vector.latency_ms if vector else None,
            "hybrid_latency_ms": hybrid.latency_ms,
            "result_count": len(chunks),
        },
        "orchestration": {
            "mode": mode,
            "hop_count": hop_count,
            "repair_count": repair_count,
            "repair_outcome": repair_outcome,
            "searched_queries": searched_queries,
        },
        "verification": {
            "abstain": verification.abstain if verification else None,
            "verdict_code": verification.verdict_code if verification else None,
            "confidence": verification.confidence if verification else None,
            "decision_path": verification.decision_path if verification else [],
            "issues": [issue.type for issue in verification.issues]
            if verification
            else [],
            "conflicts": verification.conflicts if verification else [],
            "stage_timing_ms": verifier_timing_ms,
        },
    }

    return OrchestrationResult(
        mode=mode,
        intent=decision.primary_intent.value,
        chunks=chunks,
        draft_answer=draft_answer,
        verification=verification,
        tool_trace=tool_trace,
        latency_ms=elapsed_ms,
    )
