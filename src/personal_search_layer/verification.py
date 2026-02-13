"""Answer verification, conflict detection, and deterministic repair."""

from __future__ import annotations

import re

from personal_search_layer.answering import synthesize_extractive
from personal_search_layer.models import (
    DraftAnswer,
    ScoredChunk,
    VerificationIssue,
    VerificationResult,
)
from personal_search_layer.router import PrimaryIntent, VerifierMode

_NUMBER_FACT_RE = re.compile(
    r"\b([a-z][a-z0-9\s_-]{2,40})\s+(?:is|are|was|were)\s+([0-9]{1,4})\b",
    re.IGNORECASE,
)
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _claim_supported(claim_text: str, chunk_text: str) -> bool:
    claim_tokens = [
        token
        for token in re.findall(r"[a-z0-9]+", claim_text.lower())
        if len(token) > 2
    ]
    if not claim_tokens:
        return False
    chunk_lower = chunk_text.lower()
    overlap = sum(1 for token in claim_tokens if token in chunk_lower)
    critical_tokens = [
        token for token in claim_tokens if len(token) >= 6 or token.isdigit()
    ]
    if critical_tokens and any(token not in chunk_lower for token in critical_tokens):
        return False
    return overlap >= max(2, int(len(claim_tokens) * 0.6))


def _detect_conflicts(chunks: list[ScoredChunk]) -> list[str]:
    facts: dict[str, dict[str, set[str]]] = {}
    for chunk in chunks:
        for match in _NUMBER_FACT_RE.finditer(chunk.chunk_text):
            subject = " ".join(match.group(1).lower().split())
            value = match.group(2)
            subject_map = facts.setdefault(subject, {})
            subject_map.setdefault(value, set()).add(chunk.source_path)

    conflicts: list[str] = []
    for subject, values in facts.items():
        if len(values) <= 1:
            continue
        value_parts = []
        for value, sources in sorted(values.items()):
            src_list = ", ".join(sorted(sources))
            value_parts.append(f"{value} ({src_list})")
        conflicts.append(f"Conflict for '{subject}': " + " vs ".join(value_parts))
    return conflicts


def verify_answer(
    query: str,
    draft: DraftAnswer,
    chunks: list[ScoredChunk],
    mode: VerifierMode,
) -> VerificationResult:
    issues: list[VerificationIssue] = []
    if mode == VerifierMode.OFF:
        return VerificationResult(
            passed=True,
            issues=[],
            conflicts=[],
            abstain=False,
            abstain_reason=None,
            searched_queries=list(draft.searched_queries),
        )

    query_tokens = {
        token
        for token in _TOKEN_RE.findall(query.lower())
        if len(token) >= 4 and token not in {"what", "when", "where", "which", "with", "that"}
    }
    claim_alignment = 0
    chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    for claim in draft.claims:
        claim_tokens = set(_TOKEN_RE.findall(claim.text.lower()))
        if len(claim_tokens & query_tokens) >= 2:
            claim_alignment += 1
        if not claim.citations:
            issues.append(
                VerificationIssue(
                    type="missing_citation",
                    claim_id=claim.claim_id,
                    detail="Claim has no citations.",
                )
            )
            continue
        supported = False
        for citation in claim.citations:
            chunk = chunk_by_id.get(citation.chunk_id)
            if not chunk:
                continue
            if _claim_supported(claim.text, chunk.chunk_text):
                supported = True
                break
        if not supported:
            issues.append(
                VerificationIssue(
                    type="unsupported_claim",
                    claim_id=claim.claim_id,
                    detail=claim.text,
                )
            )

    conflicts = (
        _detect_conflicts(chunks)
        if mode in {VerifierMode.STRICT, VerifierMode.STRICT_CONFLICT}
        else []
    )
    abstain = False
    abstain_reason: str | None = None

    if not draft.claims:
        abstain = True
        abstain_reason = (
            "No grounded claims could be extracted from retrieved evidence."
        )
    elif query_tokens and claim_alignment == 0:
        issues.append(
            VerificationIssue(
                type="query_mismatch",
                claim_id=None,
                detail="Retrieved claims are not aligned with the query topic.",
            )
        )
        abstain = True
        abstain_reason = "Retrieved evidence did not match the query topic."
    elif any(
        issue.type in {"missing_citation", "unsupported_claim"} for issue in issues
    ):
        abstain = True
        abstain_reason = "Retrieved evidence did not fully support all claims."
    elif mode == VerifierMode.STRICT_CONFLICT and conflicts:
        abstain = True
        abstain_reason = "Conflicting evidence detected in retrieved sources."

    passed = not abstain
    return VerificationResult(
        passed=passed,
        issues=issues,
        conflicts=conflicts,
        abstain=abstain,
        abstain_reason=abstain_reason,
        searched_queries=list(draft.searched_queries),
    )


def repair_answer(
    query: str,
    draft: DraftAnswer,
    chunks: list[ScoredChunk],
    mode: VerifierMode,
    *,
    intent: PrimaryIntent,
) -> DraftAnswer | None:
    """Attempt a single deterministic repair by re-synthesizing from available chunks."""
    verification = verify_answer(query, draft, chunks, mode)
    has_unsupported = any(
        issue.type in {"missing_citation", "unsupported_claim"}
        for issue in verification.issues
    )
    if not has_unsupported:
        return draft
    repaired = synthesize_extractive(query, chunks, intent)
    repaired.searched_queries = list(draft.searched_queries)
    repaired_verification = verify_answer(query, repaired, chunks, mode)
    if repaired_verification.passed:
        return repaired
    return None
