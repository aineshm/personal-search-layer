import os
import subprocess
import sys
from pathlib import Path


def test_query_cli_answer_mode_outputs_citations(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = tmp_path / "data"
    env = os.environ.copy()
    env["PSL_DATA_DIR"] = str(data_dir)
    env["PYTHONPATH"] = str(repo_root / "src")

    ingest = subprocess.run(
        [
            sys.executable,
            "scripts/ingest.py",
            "--path",
            "reference_docs/smoke_corpus",
            "--chunk-size",
            "200",
            "--chunk-overlap",
            "20",
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
    )
    assert ingest.returncode == 0, ingest.stderr

    query = subprocess.run(
        [
            sys.executable,
            "scripts/query.py",
            "smoke corpus keyword",
            "--mode",
            "answer",
            "--skip-vector",
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
    )
    assert query.returncode == 0, query.stderr
    assert "Claims and citations:" in query.stdout
    assert "citation chunk=" in query.stdout
