from pathlib import Path

from personal_search_layer.ingestion.pipeline import _collect_files


def test_collect_files_filters_suffixes(tmp_path: Path) -> None:
    (tmp_path / "note.txt").write_text("hello")
    (tmp_path / "skip.bin").write_bytes(b"abc")
    files = _collect_files(tmp_path)
    assert len(files) == 1
    assert files[0].suffix == ".txt"


def test_collect_files_excludes_blocked_suffixes(tmp_path: Path) -> None:
    (tmp_path / "data.json").write_text("{\"a\": 1}")
    (tmp_path / "note.txt").write_text("hello")
    files = _collect_files(tmp_path, exclude_suffixes={".json"})
    assert len(files) == 1
    assert files[0].suffix == ".txt"
