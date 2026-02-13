"""Human-readable summary for golden eval reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _format_value(value: float | int | None) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.3f}"
    return "n/a"


def _format_delta(value: float | int | None) -> str:
    if isinstance(value, (int, float)):
        return f"{value:+.3f}"
    return "n/a"


def render_summary(report: dict) -> str:
    lines: list[str] = []
    lines.append("Evaluation summary")
    lines.append(
        f"cases: {report.get('cases', 'n/a')} | top_k_default: {report.get('top_k_default', 'n/a')}"
    )
    lines.append(
        "model: {name} | revision: {rev} | commit: {commit}".format(
            name=report.get("model_name", "n/a"),
            rev=report.get("model_revision", "n/a"),
            commit=report.get("git_commit", "n/a"),
        )
    )
    router = report.get("router_accuracy")
    if isinstance(router, dict) and "accuracy" in router:
        lines.append(
            "router accuracy: {acc} ({correct}/{total})".format(
                acc=_format_value(router.get("accuracy")),
                correct=router.get("correct", "n/a"),
                total=router.get("total", "n/a"),
            )
        )
    lines.append("")
    header = (
        f"{'mode':<8} {'recall':>7} {'mrr':>7} {'ndcg':>7}"
        f" {'Δrecall':>8} {'Δmrr':>7} {'Δndcg':>7}"
    )
    lines.append(header)
    metrics_by_mode = report.get("metrics@k", {})
    deltas_by_mode = report.get("metrics_delta", {})
    for mode in ("lexical", "vector", "hybrid"):
        metrics = metrics_by_mode.get(mode, {}) if isinstance(metrics_by_mode, dict) else {}
        deltas = deltas_by_mode.get(mode, {}) if isinstance(deltas_by_mode, dict) else {}
        line = (
            f"{mode:<8}"
            f" {_format_value(metrics.get('recall')):>7}"
            f" {_format_value(metrics.get('mrr')):>7}"
            f" {_format_value(metrics.get('ndcg')):>7}"
            f" {_format_delta(deltas.get('recall')):>8}"
            f" {_format_delta(deltas.get('mrr')):>7}"
            f" {_format_delta(deltas.get('ndcg')):>7}"
        )
        lines.append(line)
    return "\n".join(lines)


def _load_report(path: Path) -> dict:
    return json.loads(path.read_text())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize eval report JSON")
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("eval/reports/latest.json"),
        help="Path to eval report JSON",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = _load_report(args.report_path)
    print(render_summary(report))


if __name__ == "__main__":
    main()
