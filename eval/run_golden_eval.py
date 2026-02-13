"""Run golden retrieval evaluation for lexical, vector, and hybrid modes."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from personal_search_layer.config import MODEL_NAME, MODEL_REVISION
from personal_search_layer.indexing import build_vector_index
from personal_search_layer.retrieval import fuse_hybrid, search_lexical, search_vector
from personal_search_layer.router import route_query


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run golden retrieval evaluation")
    parser.add_argument(
        "--cases",
        type=Path,
        default=Path("eval/golden_retrieval.jsonl"),
        help="Path to golden retrieval cases (jsonl)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Top-k cutoff for recall",
    )
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="Rebuild FAISS index before evaluation",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("eval/reports/latest.json"),
        help="Path to write eval report artifact",
    )
    parser.add_argument(
        "--history-dir",
        type=Path,
        default=Path("eval/reports/history"),
        help="Directory for timestamped report history",
    )
    return parser.parse_args()
def _load_previous_report(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def _compute_deltas(current: dict, previous: dict | None) -> dict | None:
    if not previous:
        return None
    current_metrics = current.get("metrics@k")
    previous_metrics = previous.get("metrics@k")
    if not isinstance(current_metrics, dict) or not isinstance(previous_metrics, dict):
        return None
    deltas: dict[str, dict[str, float]] = {}
    for mode, metrics in current_metrics.items():
        if not isinstance(metrics, dict):
            continue
        prev = previous_metrics.get(mode, {}) if isinstance(previous_metrics, dict) else {}
        deltas[mode] = {}
        for metric, value in metrics.items():
            if not isinstance(value, (int, float)):
                continue
            prev_value = prev.get(metric)
            if isinstance(prev_value, (int, float)):
                deltas[mode][metric] = value - prev_value
    return deltas


def load_cases(path: Path) -> list[dict]:
    cases: list[dict] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        cases.append(json.loads(line))
    return cases


def load_router_cases(path: Path) -> list[dict]:
    if not path.exists():
        return []
    cases: list[dict] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        cases.append(json.loads(line))
    return cases


def _expected_set(expected_sources: list[str]) -> list[str]:
    return [src.lower() for src in expected_sources]


def recall_at_k(chunks: list, expected_sources: list[str]) -> float:
    if not expected_sources:
        return 0.0
    expected = _expected_set(expected_sources)
    return float(
        any(
            any(chunk.source_path.lower().endswith(src) for src in expected)
            for chunk in chunks
        )
    )


def mrr_at_k(chunks: list, expected_sources: list[str]) -> float:
    if not expected_sources:
        return 0.0
    expected = _expected_set(expected_sources)
    for rank, chunk in enumerate(chunks, start=1):
        if any(chunk.source_path.lower().endswith(src) for src in expected):
            return 1.0 / rank
    return 0.0


def ndcg_at_k(chunks: list, expected_sources: list[str]) -> float:
    if not expected_sources:
        return 0.0
    expected = _expected_set(expected_sources)
    for rank, chunk in enumerate(chunks, start=1):
        if any(chunk.source_path.lower().endswith(src) for src in expected):
            return 1.0 / (1.0 + math.log2(rank + 1))
    return 0.0


def _get_git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return result.stdout.strip() or None


def _router_accuracy(cases: list[dict]) -> dict[str, float] | None:
    if not cases:
        return None
    correct = 0
    for case in cases:
        query = case.get("query", "")
        expected = case.get("intent")
        if not query or not expected:
            continue
        predicted = route_query(query).primary_intent.value
        if predicted == expected:
            correct += 1
    total = len([case for case in cases if case.get("query") and case.get("intent")])
    if total == 0:
        return None
    return {"correct": correct, "total": total, "accuracy": correct / total}


def main() -> None:
    args = parse_args()
    cases = load_cases(args.cases)
    router_cases = load_router_cases(Path("eval/router_intents.jsonl"))
    if args.rebuild_index:
        build_vector_index(model_name=MODEL_NAME, backend="sentence-transformers")

    totals = {
        "lexical": {"recall": 0.0, "mrr": 0.0, "ndcg": 0.0},
        "vector": {"recall": 0.0, "mrr": 0.0, "ndcg": 0.0},
        "hybrid": {"recall": 0.0, "mrr": 0.0, "ndcg": 0.0},
    }
    per_intent: dict[str, dict[str, dict[str, float]]] = {}
    per_case: list[dict[str, object]] = []
    for case in cases:
        query = case["query"]
        expected = case.get("expected_sources", [])
        top_k = int(case.get("top_k", args.top_k))
        intent = route_query(query).primary_intent.value
        lexical = search_lexical(query, k=top_k)
        vector = search_vector(
            query,
            k=top_k,
            backend="sentence-transformers",
            model_name=MODEL_NAME,
        )
        hybrid = fuse_hybrid(lexical, vector, k=top_k)
        lexical_metrics = {
            "recall": recall_at_k(lexical.chunks, expected),
            "mrr": mrr_at_k(lexical.chunks, expected),
            "ndcg": ndcg_at_k(lexical.chunks, expected),
        }
        vector_metrics = {
            "recall": recall_at_k(vector.chunks, expected),
            "mrr": mrr_at_k(vector.chunks, expected),
            "ndcg": ndcg_at_k(vector.chunks, expected),
        }
        hybrid_metrics = {
            "recall": recall_at_k(hybrid.chunks, expected),
            "mrr": mrr_at_k(hybrid.chunks, expected),
            "ndcg": ndcg_at_k(hybrid.chunks, expected),
        }
        for mode, metrics in {
            "lexical": lexical_metrics,
            "vector": vector_metrics,
            "hybrid": hybrid_metrics,
        }.items():
            totals[mode]["recall"] += metrics["recall"]
            totals[mode]["mrr"] += metrics["mrr"]
            totals[mode]["ndcg"] += metrics["ndcg"]
            per_intent.setdefault(intent, {"lexical": {"recall": 0.0, "mrr": 0.0, "ndcg": 0.0},
                                           "vector": {"recall": 0.0, "mrr": 0.0, "ndcg": 0.0},
                                           "hybrid": {"recall": 0.0, "mrr": 0.0, "ndcg": 0.0}})
            per_intent[intent][mode]["recall"] += metrics["recall"]
            per_intent[intent][mode]["mrr"] += metrics["mrr"]
            per_intent[intent][mode]["ndcg"] += metrics["ndcg"]

        per_case.append(
            {
                "query": query,
                "intent": intent,
                "top_k": top_k,
                "metrics": {
                    "lexical": lexical_metrics,
                    "vector": vector_metrics,
                    "hybrid": hybrid_metrics,
                },
            }
        )

    count = max(len(cases), 1)
    summary = {
        mode: {metric: value / count for metric, value in metrics.items()}
        for mode, metrics in totals.items()
    }
    intent_summary: dict[str, dict[str, dict[str, float]]] = {}
    for intent, metrics_by_mode in per_intent.items():
        intent_count = sum(1 for case in per_case if case["intent"] == intent)
        if intent_count == 0:
            continue
        intent_summary[intent] = {
            mode: {
                metric: value / intent_count for metric, value in metrics.items()
            }
            for mode, metrics in metrics_by_mode.items()
        }
    report = {
        "cases": count,
        "metrics@k": summary,
        "metrics_by_intent": intent_summary,
        "cases_detail": per_case,
        "router_accuracy": _router_accuracy(router_cases),
        "backend": "sentence-transformers",
        "model_name": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "top_k_default": args.top_k,
        "git_commit": _get_git_commit(),
    }
    previous = _load_previous_report(args.report_path)
    deltas = _compute_deltas(report, previous)
    if deltas:
        report["metrics_delta"] = deltas

    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(json.dumps(report, indent=2))

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    args.history_dir.mkdir(parents=True, exist_ok=True)
    history_path = args.history_dir / f"report_{timestamp}.json"
    history_path.write_text(json.dumps(report, indent=2))

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
