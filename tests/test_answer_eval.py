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
    assert "metrics" in report
    assert "citation_coverage" in report["metrics"]
    assert "abstain_correctness" in report["metrics"]
    assert "conflict_correctness" in report["metrics"]
    assert isinstance(report.get("cases_detail"), list)
