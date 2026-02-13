from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_run_answer_eval_outputs_expected_schema(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    report_path = tmp_path / "answer_latest.json"
    history_dir = tmp_path / "history"

    result = subprocess.run(
        [
            sys.executable,
            "eval/run_answer_eval.py",
            "--report-path",
            str(report_path),
            "--history-dir",
            str(history_dir),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    report = json.loads(report_path.read_text())
    assert report["schema_version"] == "3.0"
    assert "metrics" in report
    assert "citation_coverage" in report["metrics"]
    assert "abstain_correctness" in report["metrics"]
    assert "conflict_correctness" in report["metrics"]
    assert "false_answer_rate" in report["metrics"]
    assert "metrics_by_intent" in report
    assert "metrics_by_case_family" in report
    assert "gates" in report
    assert "hard" in report["gates"]
    assert "soft" in report["gates"]
    assert "hard_pass" in report["gates"]
    assert "overall_pass" in report["gates"]
    assert isinstance(report.get("cases_detail"), list)


def test_run_answer_eval_supports_explicit_baseline(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        json.dumps(
            {
                "metrics": {
                    "citation_coverage": 0.1,
                    "citation_precision_proxy": 0.1,
                    "abstain_correctness": 0.1,
                    "conflict_correctness": 0.1,
                    "repair_rate": 0.0,
                    "false_repair_rate": 0.0,
                    "false_answer_rate": 0.1,
                    "over_abstain_rate": 0.1,
                    "under_abstain_rate": 0.1,
                    "unsupported_claim_rate": 0.1,
                }
            }
        )
    )
    report_path = tmp_path / "answer_latest.json"
    history_dir = tmp_path / "history"

    result = subprocess.run(
        [
            sys.executable,
            "eval/run_answer_eval.py",
            "--baseline-path",
            str(baseline),
            "--report-path",
            str(report_path),
            "--history-dir",
            str(history_dir),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    report = json.loads(report_path.read_text())
    assert report.get("baseline_path") == str(baseline)
    assert "metrics_delta" in report
