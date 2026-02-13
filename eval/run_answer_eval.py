"""Run answer-mode verification evaluation and emit report artifacts."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from personal_search_layer.answering import synthesize_extractive
from personal_search_layer.models import DraftAnswer, ScoredChunk, VerificationResult
from personal_search_layer.orchestration import run_query
from personal_search_layer.router import route_query
from personal_search_layer.verification import verify_answer


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
        if issue.claim_id and issue.type in {"missing_citation", "unsupported_claim"}
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
    )
    verification.searched_queries = [query]
    tool_trace = {
        "orchestration": {
            "hop_count": 0,
            "repair_count": 0,
            "searched_queries": [query],
        }
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
    }

    details: list[dict] = []
    repairs = 0
    false_repairs = 0
    for case in cases:
        if case.get("synthetic_chunks"):
            draft, verification, tool_trace = _run_synthetic_case(case)
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

        expected_abstain = bool(case.get("expected_abstain", False))
        expect_conflict = bool(case.get("expect_conflict", False))
        actual_abstain = bool(verification.abstain) if verification else True
        actual_conflict = bool(verification and verification.conflicts)

        coverage = _citation_coverage(draft)
        precision = _citation_precision_proxy(draft, verification)

        metrics["citation_coverage"] += coverage
        metrics["citation_precision_proxy"] += precision
        metrics["abstain_correctness"] += float(actual_abstain == expected_abstain)
        metrics["conflict_correctness"] += float(actual_conflict == expect_conflict)

        repair_count = int(tool_trace.get("orchestration", {}).get("repair_count", 0))
        if repair_count > 0:
            repairs += 1
            if actual_abstain:
                false_repairs += 1

        details.append(
            {
                "id": case.get("id"),
                "query": case.get("query"),
                "expected_abstain": expected_abstain,
                "actual_abstain": actual_abstain,
                "expect_conflict": expect_conflict,
                "actual_conflict": actual_conflict,
                "citation_coverage": coverage,
                "citation_precision_proxy": precision,
                "repair_count": repair_count,
            }
        )

    metrics = {key: value / total for key, value in metrics.items()}
    metrics["repair_rate"] = repairs / total
    metrics["false_repair_rate"] = (false_repairs / repairs) if repairs else 0.0

    report = {
        "cases": len(cases),
        "metrics": metrics,
        "thresholds": {
            "citation_coverage_min": 0.98,
            "abstain_correctness_min": 0.95,
            "conflict_correctness_min": 0.85,
        },
        "cases_detail": details,
    }
    report["gates"] = {
        "citation_coverage_pass": metrics["citation_coverage"] >= 0.98,
        "abstain_correctness_pass": metrics["abstain_correctness"] >= 0.95,
        "conflict_correctness_pass": metrics["conflict_correctness"] >= 0.85,
        "overall_pass": (
            metrics["citation_coverage"] >= 0.98
            and metrics["abstain_correctness"] >= 0.95
            and metrics["conflict_correctness"] >= 0.85
        ),
    }

    previous = _load_previous(args.report_path)
    deltas = _compute_deltas(report, previous)
    if deltas:
        report["metrics_delta"] = deltas

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
