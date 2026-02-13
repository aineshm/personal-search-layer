"""Lightweight query router for intent classification.

Provides deterministic primary intent labels plus recommended pipeline settings
that downstream orchestration can enforce (k, lexical weighting, rerank, and
bounded multi-hop/repair settings).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable


class PrimaryIntent(str, Enum):
    LOOKUP = "lookup"
    FACT = "fact"
    SYNTHESIS = "synthesis"
    COMPARE = "compare"
    TIMELINE = "timeline"
    TASK = "task"
    OTHER = "other"


class VerifierMode(str, Enum):
    OFF = "off"
    MINIMAL = "minimal"
    STRICT = "strict"
    STRICT_CONFLICT = "strict_conflict"


@dataclass(frozen=True)
class IntentFlags:
    wants_definition: bool = False
    wants_steps: bool = False
    wants_summary: bool = False


@dataclass(frozen=True)
class PipelineSettings:
    k: int
    lexical_weight: float
    allow_multihop: int
    use_rerank: bool
    generate_answer: bool
    verifier_mode: VerifierMode
    max_repair_passes: int = 1


@dataclass(frozen=True)
class RouteDecision:
    primary_intent: PrimaryIntent
    flags: IntentFlags
    recommended_pipeline_settings: PipelineSettings
    signals: list[str]


def _contains_any(text: str, phrases: Iterable[str]) -> bool:
    return any(phrase in text for phrase in phrases)


def _detect_flags(normalized: str, signals: list[str]) -> IntentFlags:
    wants_definition = _contains_any(normalized, ["what is", "define", "definition"])
    wants_steps = _contains_any(normalized, ["how to", "steps", "procedure", "guide", "how do i"])
    wants_summary = _contains_any(normalized, ["summary", "summarize", "overview"])
    if wants_definition:
        signals.append("definition_phrase")
    if wants_steps:
        signals.append("steps_phrase")
    if wants_summary:
        signals.append("summary_phrase")
    return IntentFlags(
        wants_definition=wants_definition,
        wants_steps=wants_steps,
        wants_summary=wants_summary,
    )


def _classify_primary_intent(
    normalized: str, flags: IntentFlags, signals: list[str]
) -> PrimaryIntent:
    if not normalized:
        return PrimaryIntent.OTHER
    if "\"" in normalized or _contains_any(normalized, ["exact", "verbatim", "quote"]):
        signals.append("explicit_lookup")
        return PrimaryIntent.LOOKUP
    if _contains_any(normalized, ["compare", "difference", "diff", "vs", "versus"]):
        signals.append("compare_phrase")
        return PrimaryIntent.COMPARE
    if _contains_any(normalized, ["timeline", "chronology", "milestones", "dates"]):
        signals.append("timeline_phrase")
        return PrimaryIntent.TIMELINE
    if flags.wants_steps or _contains_any(
        normalized, ["checklist", "plan", "todo", "tasks", "steps to"]
    ):
        signals.append("task_phrase")
        return PrimaryIntent.TASK
    if flags.wants_summary or _contains_any(
        normalized, ["combine", "synthesize", "across sources", "overall", "merge"]
    ):
        signals.append("synthesis_phrase")
        return PrimaryIntent.SYNTHESIS
    if flags.wants_definition or normalized.endswith("?") or _contains_any(
        normalized, ["who", "when", "where", "which", "what", "how many"]
    ):
        signals.append("fact_phrase")
        return PrimaryIntent.FACT
    if len(normalized.split()) <= 4:
        signals.append("short_query")
        return PrimaryIntent.LOOKUP
    return PrimaryIntent.OTHER


def default_pipeline_settings(intent: PrimaryIntent) -> PipelineSettings:
    if intent == PrimaryIntent.LOOKUP:
        return PipelineSettings(
            k=8,
            lexical_weight=0.8,
            allow_multihop=0,
            use_rerank=False,
            generate_answer=False,
            verifier_mode=VerifierMode.MINIMAL,
            max_repair_passes=0,
        )
    if intent == PrimaryIntent.FACT:
        return PipelineSettings(
            k=10,
            lexical_weight=0.5,
            allow_multihop=0,
            use_rerank=False,
            generate_answer=True,
            verifier_mode=VerifierMode.STRICT,
            max_repair_passes=1,
        )
    if intent == PrimaryIntent.SYNTHESIS:
        return PipelineSettings(
            k=24,
            lexical_weight=0.4,
            allow_multihop=1,
            use_rerank=True,
            generate_answer=True,
            verifier_mode=VerifierMode.STRICT_CONFLICT,
            max_repair_passes=1,
        )
    if intent == PrimaryIntent.COMPARE:
        return PipelineSettings(
            k=20,
            lexical_weight=0.5,
            allow_multihop=1,
            use_rerank=True,
            generate_answer=True,
            verifier_mode=VerifierMode.STRICT,
            max_repair_passes=1,
        )
    if intent == PrimaryIntent.TIMELINE:
        return PipelineSettings(
            k=20,
            lexical_weight=0.6,
            allow_multihop=1,
            use_rerank=True,
            generate_answer=True,
            verifier_mode=VerifierMode.STRICT_CONFLICT,
            max_repair_passes=1,
        )
    if intent == PrimaryIntent.TASK:
        return PipelineSettings(
            k=20,
            lexical_weight=0.4,
            allow_multihop=1,
            use_rerank=True,
            generate_answer=True,
            verifier_mode=VerifierMode.STRICT,
            max_repair_passes=1,
        )
    return PipelineSettings(
        k=12,
        lexical_weight=0.5,
        allow_multihop=0,
        use_rerank=False,
        generate_answer=True,
        verifier_mode=VerifierMode.STRICT,
        max_repair_passes=1,
    )


def route_query(query: str) -> RouteDecision:
    normalized = query.strip().lower()
    signals: list[str] = []
    flags = _detect_flags(normalized, signals)
    primary_intent = _classify_primary_intent(normalized, flags, signals)
    settings = default_pipeline_settings(primary_intent)
    return RouteDecision(
        primary_intent=primary_intent,
        flags=flags,
        recommended_pipeline_settings=settings,
        signals=signals,
    )
