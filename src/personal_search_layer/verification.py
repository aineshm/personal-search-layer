"""Answer verification, conflict detection, and deterministic repair."""

from __future__ import annotations

import re

from personal_search_layer.answering import synthesize_extractive
from personal_search_layer.config import (
    VERIFIER_AGGREGATE_MIN,
    VERIFIER_CITATION_SPAN_QUALITY_MIN,
    VERIFIER_CLAIM_SUPPORT_MIN,
    VERIFIER_CRITICAL_COVERAGE_MIN,
    VERIFIER_QUERY_ALIGNMENT_MIN,
)
from personal_search_layer.models import (
    DraftAnswer,
    ScoredChunk,
    VerificationIssue,
    VerificationResult,
)
from personal_search_layer.router import PrimaryIntent, VerifierMode

_NUMBER_FACT_RE = re.compile(
    r"\b([a-z][a-z0-9\s_-]{2,40})\s+(?:is|are|was|were|has|have)\s+([0-9]{1,4})\b",
    re.IGNORECASE,
)
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "what",
    "when",
    "where",
    "which",
    "with",
    "that",
    "this",
    "from",
    "into",
    "your",
}
_PROMPT_INJECTION_TOKENS = {
    "ignore",
    "bypass",
    "safeguard",
    "safeguards",
    "environment",
    "variables",
    "unrestricted",
    "reveal",
    "password",
    "secret",
    "secrets",
    "exfil",
    "exfiltrate",
    "instructions",
}
_NON_CRITICAL_QUERY_TOKENS = {
    "mentioned",
    "mention",
    "says",
    "say",
    "describe",
    "explain",
    "summarize",
    "summary",
    "compare",
    "overview",
}
_HARD_REQUIRED_QUERY_TOKENS = {
    "retention",
    "policy",
    "encryption",
    "algorithm",
    "backup",
    "cadence",
    "database",
    "endpoint",
    "api",
}


def _token_match(token: str, text_tokens: set[str]) -> bool:
    if token in text_tokens:
        return True
    if len(token) < 5:
        return False
    return any(
        candidate.startswith(token) or token.startswith(candidate)
        for candidate in text_tokens
        if len(candidate) >= 5
    )


def _claim_supported(claim_text: str, chunk_text: str) -> float:
    claim_tokens = [
        token
        for token in _TOKEN_RE.findall(claim_text.lower())
        if len(token) > 2 and token not in _STOPWORDS
    ]
    if not claim_tokens:
        return 0.0
    chunk_lower = chunk_text.lower()
    overlap = sum(1 for token in claim_tokens if token in chunk_lower)
    critical_tokens = [
        token for token in claim_tokens if len(token) >= 6 or token.isdigit()
    ]
    if critical_tokens and any(token not in chunk_lower for token in critical_tokens):
        return 0.0
    return overlap / len(claim_tokens)


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


def _query_tokens(query: str) -> set[str]:
    return {
        token
        for token in _TOKEN_RE.findall(query.lower())
        if len(token) >= 4 and token not in _STOPWORDS
    }


def _contains_prompt_injection_signal(query_tokens: set[str]) -> bool:
    return any(token in _PROMPT_INJECTION_TOKENS for token in query_tokens)


def _critical_coverage_min(intent: PrimaryIntent | None) -> float:
    # Facts need stricter entity/term coverage than synthesis-style intents.
    if intent == PrimaryIntent.FACT:
        return max(VERIFIER_CRITICAL_COVERAGE_MIN, 0.5)
    if intent in {
        PrimaryIntent.SYNTHESIS,
        PrimaryIntent.COMPARE,
        PrimaryIntent.TIMELINE,
        PrimaryIntent.TASK,
        PrimaryIntent.OTHER,
    }:
        return min(VERIFIER_CRITICAL_COVERAGE_MIN, 0.2)
    return VERIFIER_CRITICAL_COVERAGE_MIN


def _required_alignment_overlap(
    intent: PrimaryIntent | None, query_token_count: int
) -> int:
    # Synthesis/compare prompts often map to broader language than fact lookups.
    if query_token_count <= 1:
        return 1
    if intent in {
        PrimaryIntent.SYNTHESIS,
        PrimaryIntent.COMPARE,
        PrimaryIntent.TIMELINE,
        PrimaryIntent.TASK,
        PrimaryIntent.OTHER,
    }:
        return 1
    return 2


def verify_answer(
    query: str,
    draft: DraftAnswer,
    chunks: list[ScoredChunk],
    mode: VerifierMode,
    *,
    intent: PrimaryIntent | None = None,
) -> VerificationResult:
    issues: list[VerificationIssue] = []
    decision_path: list[str] = []
    query_tokens = _query_tokens(query)

    # Treat jailbreak-like requests as mismatches even when no claims are extracted.
    if _contains_prompt_injection_signal(query_tokens):
        return VerificationResult(
            passed=False,
            issues=[
                VerificationIssue(
                    type="query_mismatch",
                    claim_id=None,
                    detail="Prompt-injection-like request is unsupported in evidence-only mode.",
                )
            ],
            conflicts=[],
            abstain=True,
            abstain_reason="Request is not answerable from trusted corpus evidence.",
            verdict_code="query_mismatch",
            confidence=0.0,
            decision_path=["prompt_injection_signal"],
            searched_queries=list(draft.searched_queries),
        )

    if mode == VerifierMode.OFF:
        return VerificationResult(
            passed=True,
            issues=[],
            conflicts=[],
            abstain=False,
            abstain_reason=None,
            verdict_code="supported",
            confidence=1.0,
            decision_path=["mode_off"],
            searched_queries=list(draft.searched_queries),
        )

    if not draft.claims:
        return VerificationResult(
            passed=False,
            issues=[
                VerificationIssue(
                    type="insufficient_evidence",
                    claim_id=None,
                    detail="No claims were available for verification.",
                )
            ],
            conflicts=[],
            abstain=True,
            abstain_reason="No grounded claims could be extracted from retrieved evidence.",
            verdict_code="insufficient_evidence",
            confidence=0.0,
            decision_path=["no_claims"],
            searched_queries=list(draft.searched_queries),
        )

    chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    all_claim_tokens: set[str] = set()

    aligned_claims = 0
    supported_claims = 0
    citation_ok_claims = 0

    for claim in draft.claims:
        claim_tokens = set(_TOKEN_RE.findall(claim.text.lower()))
        all_claim_tokens |= claim_tokens
        required_overlap = _required_alignment_overlap(intent, len(query_tokens))
        overlap_count = sum(
            1 for token in query_tokens if _token_match(token, claim_tokens)
        )
        if query_tokens and overlap_count >= required_overlap:
            aligned_claims += 1

        if not claim.citations:
            issues.append(
                VerificationIssue(
                    type="citation_gap",
                    claim_id=claim.claim_id,
                    detail="Claim has no citations.",
                )
            )
            continue

        span_quality = max(
            citation.quote_span_end - citation.quote_span_start
            for citation in claim.citations
        ) / max(1, len(claim.text))
        if (
            max(claim.citation_span_quality, span_quality)
            >= VERIFIER_CITATION_SPAN_QUALITY_MIN
        ):
            citation_ok_claims += 1
        else:
            issues.append(
                VerificationIssue(
                    type="citation_gap",
                    claim_id=claim.claim_id,
                    detail="Citation spans were too weak for this claim.",
                )
            )

        claim_supported = False
        claim_best_support = 0.0
        for citation in claim.citations:
            chunk = chunk_by_id.get(citation.chunk_id)
            if not chunk:
                continue
            support_score = _claim_supported(claim.text, chunk.chunk_text)
            claim_best_support = max(claim_best_support, support_score)
            if support_score >= VERIFIER_CLAIM_SUPPORT_MIN:
                claim_supported = True
                break
        if claim_supported:
            supported_claims += 1
        else:
            issues.append(
                VerificationIssue(
                    type="unsupported_claim",
                    claim_id=claim.claim_id,
                    detail=f"{claim.text} (support={claim_best_support:.2f})",
                )
            )

    claim_total = max(1, len(draft.claims))
    query_alignment_score = aligned_claims / claim_total
    claim_support_score = supported_claims / claim_total
    citation_span_quality_score = citation_ok_claims / claim_total
    critical_query_tokens = {
        token
        for token in query_tokens
        if (len(token) >= 6 or token.isdigit())
        and token not in _NON_CRITICAL_QUERY_TOKENS
    }
    missing_critical_tokens = {
        token for token in critical_query_tokens if not _token_match(token, all_claim_tokens)
    }
    critical_coverage_score = (
        sum(
            1
            for token in critical_query_tokens
            if _token_match(token, all_claim_tokens)
        )
        / max(1, len(critical_query_tokens))
        if critical_query_tokens
        else 1.0
    )

    conflicts = (
        _detect_conflicts(chunks)
        if mode in {VerifierMode.STRICT, VerifierMode.STRICT_CONFLICT}
        else []
    )
    agreement_score = 0.0 if conflicts else 1.0

    if query_tokens and query_alignment_score < VERIFIER_QUERY_ALIGNMENT_MIN:
        decision_path.append("query_alignment_failed")
        issues.append(
            VerificationIssue(
                type="query_mismatch",
                claim_id=None,
                detail="Retrieved claims are not aligned with the query topic.",
            )
        )
        return VerificationResult(
            passed=False,
            issues=issues,
            conflicts=conflicts,
            abstain=True,
            abstain_reason="Retrieved evidence did not match the query topic.",
            verdict_code="query_mismatch",
            confidence=query_alignment_score,
            decision_path=decision_path,
            searched_queries=list(draft.searched_queries),
        )

    if conflicts and mode in {VerifierMode.STRICT, VerifierMode.STRICT_CONFLICT}:
        decision_path.append("conflict_detected")
        return VerificationResult(
            passed=False,
            issues=issues,
            conflicts=conflicts,
            abstain=True,
            abstain_reason="Conflicting evidence detected in retrieved sources.",
            verdict_code="conflict_detected",
            confidence=min(query_alignment_score, claim_support_score),
            decision_path=decision_path,
            searched_queries=list(draft.searched_queries),
        )

    if missing_critical_tokens & _HARD_REQUIRED_QUERY_TOKENS:
        # Hard-required tokens guard against near-miss false answers (e.g., "api endpoint").
        decision_path.append("hard_required_token_missing")
        issues.append(
            VerificationIssue(
                type="insufficient_evidence",
                claim_id=None,
                detail="Required query term(s) were not supported by retrieved claims: "
                + ", ".join(sorted(missing_critical_tokens & _HARD_REQUIRED_QUERY_TOKENS)),
            )
        )
        return VerificationResult(
            passed=False,
            issues=issues,
            conflicts=conflicts,
            abstain=True,
            abstain_reason="Evidence does not cover required query terms.",
            verdict_code="insufficient_evidence",
            confidence=critical_coverage_score,
            decision_path=decision_path,
            searched_queries=list(draft.searched_queries),
        )

    if critical_coverage_score < _critical_coverage_min(intent):
        decision_path.append("critical_token_coverage_failed")
        issues.append(
            VerificationIssue(
                type="insufficient_evidence",
                claim_id=None,
                detail="Critical query terms were not supported by retrieved claims.",
            )
        )
        return VerificationResult(
            passed=False,
            issues=issues,
            conflicts=conflicts,
            abstain=True,
            abstain_reason="Evidence does not cover the core entities/terms in the query.",
            verdict_code="insufficient_evidence",
            confidence=critical_coverage_score,
            decision_path=decision_path,
            searched_queries=list(draft.searched_queries),
        )

    if any(issue.type == "citation_gap" for issue in issues):
        decision_path.append("citation_gap")
        return VerificationResult(
            passed=False,
            issues=issues,
            conflicts=conflicts,
            abstain=True,
            abstain_reason="Citation coverage/quality was insufficient for one or more claims.",
            verdict_code="citation_gap",
            confidence=(query_alignment_score + citation_span_quality_score) / 2.0,
            decision_path=decision_path,
            searched_queries=list(draft.searched_queries),
        )

    if claim_support_score < VERIFIER_CLAIM_SUPPORT_MIN:
        decision_path.append("unsupported_claim")
        return VerificationResult(
            passed=False,
            issues=issues,
            conflicts=conflicts,
            abstain=True,
            abstain_reason="Retrieved evidence did not fully support all claims.",
            verdict_code="unsupported_claim",
            confidence=claim_support_score,
            decision_path=decision_path,
            searched_queries=list(draft.searched_queries),
        )

    aggregate_score = (
        query_alignment_score * 0.35
        + claim_support_score * 0.35
        + citation_span_quality_score * 0.20
        + agreement_score * 0.10
    )
    if aggregate_score < VERIFIER_AGGREGATE_MIN:
        decision_path.append("aggregate_below_threshold")
        return VerificationResult(
            passed=False,
            issues=issues,
            conflicts=conflicts,
            abstain=True,
            abstain_reason="Combined evidence confidence is below threshold.",
            verdict_code="insufficient_evidence",
            confidence=aggregate_score,
            decision_path=decision_path,
            searched_queries=list(draft.searched_queries),
        )

    decision_path.append("supported")
    return VerificationResult(
        passed=True,
        issues=issues,
        conflicts=conflicts,
        abstain=False,
        abstain_reason=None,
        verdict_code="supported",
        confidence=aggregate_score,
        decision_path=decision_path,
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
    verification = verify_answer(query, draft, chunks, mode, intent=intent)
    if verification.verdict_code in {
        "query_mismatch",
        "conflict_detected",
        "insufficient_evidence",
    }:
        return None
    if verification.passed:
        return draft

    repaired = synthesize_extractive(query, chunks, intent)
    repaired.searched_queries = list(draft.searched_queries)
    repaired_verification = verify_answer(query, repaired, chunks, mode, intent=intent)
    if repaired_verification.passed:
        return repaired
    return None
