"""Lightweight query router for intent classification.

Provides deterministic primary intent labels plus recommended pipeline settings
that downstream orchestration can enforce (k, lexical weighting, rerank, and
bounded multi-hop/repair settings).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable


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


def _policy_path() -> Path:
    configured = os.getenv("PSL_ROUTER_POLICY")
    if configured:
        return Path(configured).expanduser()
    return Path(__file__).resolve().with_name("router_policy.json")


@lru_cache(maxsize=1)
def _load_policy() -> dict[str, Any]:
    path = _policy_path()
    return json.loads(path.read_text())


def _detect_flags(normalized: str, signals: list[str]) -> IntentFlags:
    policy = _load_policy()["flags"]
    wants_definition = _contains_any(normalized, policy["definition"])
    wants_steps = _contains_any(normalized, policy["steps"])
    wants_summary = _contains_any(normalized, policy["summary"])
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
    policy = _load_policy()["classification"]
    if not normalized:
        return PrimaryIntent.OTHER
    if '"' in normalized or _contains_any(normalized, policy["lookup_explicit"]):
        signals.append("explicit_lookup")
        return PrimaryIntent.LOOKUP
    if _contains_any(normalized, policy["compare"]):
        signals.append("compare_phrase")
        return PrimaryIntent.COMPARE
    if _contains_any(normalized, policy["timeline"]):
        signals.append("timeline_phrase")
        return PrimaryIntent.TIMELINE
    if flags.wants_steps or _contains_any(normalized, policy["task"]):
        signals.append("task_phrase")
        return PrimaryIntent.TASK
    if flags.wants_summary or _contains_any(normalized, policy["synthesis"]):
        signals.append("synthesis_phrase")
        return PrimaryIntent.SYNTHESIS
    question_mark_is_fact = bool(policy.get("question_mark_is_fact", True))
    if (
        flags.wants_definition
        or (question_mark_is_fact and normalized.endswith("?"))
        or _contains_any(normalized, policy["fact_words"])
    ):
        signals.append("fact_phrase")
        return PrimaryIntent.FACT
    short_lookup_word_count = int(policy.get("short_lookup_word_count", 4))
    if len(normalized.split()) <= short_lookup_word_count:
        signals.append("short_query")
        return PrimaryIntent.LOOKUP
    return PrimaryIntent.OTHER


def default_pipeline_settings(intent: PrimaryIntent) -> PipelineSettings:
    policy = _load_policy()["pipeline_settings"]
    key = intent.value if intent.value in policy else "other"
    row = policy[key]
    return PipelineSettings(
        k=int(row["k"]),
        lexical_weight=float(row["lexical_weight"]),
        allow_multihop=int(row["allow_multihop"]),
        use_rerank=bool(row["use_rerank"]),
        generate_answer=bool(row["generate_answer"]),
        verifier_mode=VerifierMode(str(row["verifier_mode"])),
        max_repair_passes=int(row.get("max_repair_passes", 1)),
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
