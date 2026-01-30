import os
import subprocess
import sys
from pathlib import Path


def test_smoke_ingest_and_query(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = tmp_path / "data"
    env = os.environ.copy()
    env["PSL_DATA_DIR"] = str(data_dir)
    env["PYTHONPATH"] = str(repo_root / "src")

    ingest_cmd = [
        sys.executable,
        "scripts/ingest.py",
        "--path",
        "reference_docs/smoke_corpus",
        "--chunk-size",
        "200",
        "--chunk-overlap",
        "20",
    ]
    ingest = subprocess.run(
        ingest_cmd, cwd=repo_root, env=env, capture_output=True, text=True
    )
    assert ingest.returncode == 0, ingest.stderr

    query_cmd = [
        sys.executable,
        "scripts/query.py",
        "smoke corpus keyword",
        "--top-k",
        "5",
        "--rebuild-index",
    ]
    query = subprocess.run(
        query_cmd, cwd=repo_root, env=env, capture_output=True, text=True
    )
    assert query.returncode == 0, query.stderr
    assert "#1" in query.stdout
