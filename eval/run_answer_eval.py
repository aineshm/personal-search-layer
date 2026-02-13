"""Run answer-mode verification evaluation and emit report artifacts."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from personal_search_layer.answering import synthesize_extractive
from personal_search_layer.models import DraftAnswer, ScoredChunk, VerificationResult
from personal_search_layer.orchestration import run_query
from personal_search_layer.router import route_query
from personal_search_layer.verification import verify_answer

SCHEMA_VERSION = "2.0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run answer-mode verification eval")
    parser.add_argument(
        "--cases",
        type=Path,
        default=Path("eval/verifier_cases.jsonl"),
        help="Path to verifier/adversarial cases",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("eval/reports/answer_latest.json"),
        help="Path to write answer eval report",
    )
    parser.add_argument(
        "--history-dir",
        type=Path,
        default=Path("eval/reports/history"),
        help="Directory for timestamped answer eval reports",
    )
    parser.add_argument(
        "--baseline-path",
        type=Path,
        default=None,
        help="Optional explicit baseline report for delta and regression checks",
    )
    return parser.parse_args()


def _load_cases(path: Path) -> list[dict]:
    cases: list[dict] = []
    for line in path.read_text().splitlines():
        if line.strip():
            cases.append(json.loads(line))
    return cases


def _citation_coverage(draft: DraftAnswer | None) -> float:
    if not draft or not draft.claims:
        return 0.0
    covered = sum(1 for claim in draft.claims if claim.citations)
    return covered / len(draft.claims)


def _citation_precision_proxy(
    draft: DraftAnswer | None,
    verification: VerificationResult | None,
) -> float:
    if not draft or not draft.claims or not verification:
        return 0.0
    unsupported = {
        issue.claim_id
        for issue in verification.issues
        if issue.claim_id and issue.type in {"citation_gap", "unsupported_claim"}
    }
    supported = sum(1 for claim in draft.claims if claim.claim_id not in unsupported)
    return supported / len(draft.claims)


def _run_synthetic_case(case: dict) -> tuple[DraftAnswer, VerificationResult, dict]:
    query = case["query"]
    chunks = [ScoredChunk(**item) for item in case.get("synthetic_chunks", [])]
    route = route_query(query)
    draft = synthesize_extractive(query, chunks, route.primary_intent)
    draft.searched_queries = [query]
    verification = verify_answer(
        query,
        draft,
        chunks,
        route.recommended_pipeline_settings.verifier_mode,
        intent=route.primary_intent,
    )
    verification.searched_queries = [query]
    tool_trace = {
        "orchestration": {
            "hop_count": 0,
            "repair_count": 0,
            "repair_outcome": "none",
            "searched_queries": [query],
        },
        "verification": {
            "verdict_code": verification.verdict_code,
            "confidence": verification.confidence,
            "decision_path": verification.decision_path,
        },
    }
    return draft, verification, tool_trace


def _compute_deltas(current: dict, previous: dict | None) -> dict | None:
    if not previous:
        return None
    cur = current.get("metrics")
    prev = previous.get("metrics")
    if not isinstance(cur, dict) or not isinstance(prev, dict):
        return None
    deltas: dict[str, float] = {}
    for key, value in cur.items():
        prev_value = prev.get(key)
        if isinstance(value, (float, int)) and isinstance(prev_value, (float, int)):
            deltas[key] = float(value) - float(prev_value)
    return deltas


def _load_previous(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def _avg_rollups(
    raw: dict[str, dict[str, float]], counts: dict[str, int]
) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for key, bucket in raw.items():
        denom = max(1, counts.get(key, 0))
        output[key] = {metric: value / denom for metric, value in bucket.items()}
    return output


def main() -> None:
    args = parse_args()
    cases = _load_cases(args.cases)

    total = max(len(cases), 1)
    metrics = {
        "citation_coverage": 0.0,
        "citation_precision_proxy": 0.0,
        "abstain_correctness": 0.0,
        "conflict_correctness": 0.0,
        "repair_rate": 0.0,
        "false_repair_rate": 0.0,
        "false_answer_rate": 0.0,
        "over_abstain_rate": 0.0,
        "under_abstain_rate": 0.0,
        "unsupported_claim_rate": 0.0,
    }

    by_intent_raw: dict[str, dict[str, float]] = defaultdict(
        lambda: {
            "abstain_correctness": 0.0,
            "citation_coverage": 0.0,
            "citation_precision_proxy": 0.0,
        }
    )
    by_intent_counts: dict[str, int] = defaultdict(int)

    by_case_family_raw: dict[str, dict[str, float]] = defaultdict(
        lambda: {
            "abstain_correctness": 0.0,
            "citation_coverage": 0.0,
            "citation_precision_proxy": 0.0,
        }
    )
    by_case_family_counts: dict[str, int] = defaultdict(int)

    details: list[dict] = []
    repairs = 0
    false_repairs = 0
    for case in cases:
        if case.get("synthetic_chunks"):
            draft, verification, tool_trace = _run_synthetic_case(case)
            intent = (
                case.get("intent") or route_query(case["query"]).primary_intent.value
            )
        else:
            result = run_query(
                case["query"],
                mode="answer",
                top_k=int(case.get("top_k", 8)),
                skip_vector=True,
            )
            draft = result.draft_answer
            verification = result.verification
            tool_trace = result.tool_trace
            intent = result.intent

        case_family = str(case.get("case_family", "general"))
        risk_level = str(case.get("risk_level", "medium"))
        expected_abstain = bool(case.get("expected_abstain", False))
        expect_conflict = bool(case.get("expect_conflict", False))
        expected_verdict = str(case.get("expected_verdict", ""))
        actual_abstain = bool(verification.abstain) if verification else True
        actual_conflict = bool(verification and verification.conflicts)
        actual_verdict = (
            verification.verdict_code if verification else "insufficient_evidence"
        )

        coverage = _citation_coverage(draft)
        precision = _citation_precision_proxy(draft, verification)

        metrics["citation_coverage"] += coverage
        metrics["citation_precision_proxy"] += precision
        metrics["abstain_correctness"] += float(actual_abstain == expected_abstain)
        metrics["conflict_correctness"] += float(actual_conflict == expect_conflict)

        if expected_abstain and not actual_abstain:
            metrics["false_answer_rate"] += 1.0
            metrics["under_abstain_rate"] += 1.0
        if (not expected_abstain) and actual_abstain:
            metrics["over_abstain_rate"] += 1.0

        unsupported_present = bool(
            verification
            and any(issue.type == "unsupported_claim" for issue in verification.issues)
        )
        if unsupported_present:
            metrics["unsupported_claim_rate"] += 1.0

        by_intent_counts[intent] += 1
        by_intent_raw[intent]["abstain_correctness"] += float(
            actual_abstain == expected_abstain
        )
        by_intent_raw[intent]["citation_coverage"] += coverage
        by_intent_raw[intent]["citation_precision_proxy"] += precision

        by_case_family_counts[case_family] += 1
        by_case_family_raw[case_family]["abstain_correctness"] += float(
            actual_abstain == expected_abstain
        )
        by_case_family_raw[case_family]["citation_coverage"] += coverage
        by_case_family_raw[case_family]["citation_precision_proxy"] += precision

        repair_count = int(tool_trace.get("orchestration", {}).get("repair_count", 0))
        repair_outcome = str(
            tool_trace.get("orchestration", {}).get("repair_outcome", "none")
        )
        if repair_count > 0:
            repairs += 1
            if repair_outcome in {"harmful", "unsuccessful"}:
                false_repairs += 1

        details.append(
            {
                "id": case.get("id"),
                "query": case.get("query"),
                "intent": intent,
                "case_family": case_family,
                "risk_level": risk_level,
                "expected_abstain": expected_abstain,
                "actual_abstain": actual_abstain,
                "expected_verdict": expected_verdict,
                "actual_verdict": actual_verdict,
                "expect_conflict": expect_conflict,
                "actual_conflict": actual_conflict,
                "citation_coverage": coverage,
                "citation_precision_proxy": precision,
                "repair_count": repair_count,
                "repair_outcome": repair_outcome,
                "decision_path": verification.decision_path if verification else [],
                "confidence": verification.confidence if verification else 0.0,
            }
        )

    metrics = {key: value / total for key, value in metrics.items()}
    metrics["repair_rate"] = repairs / total
    metrics["false_repair_rate"] = (false_repairs / repairs) if repairs else 0.0

    thresholds = {
        "phase": {
            "citation_coverage_min": 0.90,
            "abstain_correctness_min": 0.95,
            "conflict_correctness_min": 0.85,
            "false_repair_rate_max": 0.20,
        },
        "long_term": {
            "citation_coverage_min": 0.98,
            "abstain_correctness_min": 0.95,
            "conflict_correctness_min": 0.85,
        },
    }

    report = {
        "schema_version": SCHEMA_VERSION,
        "cases": len(cases),
        "metrics": metrics,
        "thresholds": thresholds,
        "metrics_by_intent": _avg_rollups(by_intent_raw, by_intent_counts),
        "metrics_by_case_family": _avg_rollups(
            by_case_family_raw, by_case_family_counts
        ),
        "cases_detail": details,
    }
    report["gates"] = {
        "citation_coverage_pass": metrics["citation_coverage"]
        >= thresholds["phase"]["citation_coverage_min"],
        "abstain_correctness_pass": metrics["abstain_correctness"]
        >= thresholds["phase"]["abstain_correctness_min"],
        "conflict_correctness_pass": metrics["conflict_correctness"]
        >= thresholds["phase"]["conflict_correctness_min"],
        "false_repair_rate_pass": metrics["false_repair_rate"]
        <= thresholds["phase"]["false_repair_rate_max"],
        "overall_pass": (
            metrics["citation_coverage"] >= thresholds["phase"]["citation_coverage_min"]
            and metrics["abstain_correctness"]
            >= thresholds["phase"]["abstain_correctness_min"]
            and metrics["conflict_correctness"]
            >= thresholds["phase"]["conflict_correctness_min"]
            and metrics["false_repair_rate"]
            <= thresholds["phase"]["false_repair_rate_max"]
        ),
    }

    baseline_path = args.baseline_path or args.report_path
    previous = _load_previous(baseline_path)
    deltas = _compute_deltas(report, previous)
    if deltas:
        report["metrics_delta"] = deltas
        report["baseline_path"] = str(baseline_path)

    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(json.dumps(report, indent=2))

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    args.history_dir.mkdir(parents=True, exist_ok=True)
    (args.history_dir / f"answer_report_{timestamp}.json").write_text(
        json.dumps(report, indent=2)
    )

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
