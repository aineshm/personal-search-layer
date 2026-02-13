from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

import personal_search_layer.config as config
from personal_search_layer.storage import connect


def _load_cases(repo_root: Path) -> list[dict]:
    path = repo_root / "eval" / "golden_retrieval.jsonl"
    cases: list[dict] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        cases.append(json.loads(line))
    return cases


def _normalize_text(value: str) -> str:
    return " ".join(value.lower().split())


def _contains_source(chunks, expected_sources: list[str]) -> bool:
    expected = [src.lower() for src in expected_sources]
    return any(
        any(chunk.source_path.lower().endswith(src) for src in expected)
        for chunk in chunks
    )


def _phrase_in_sources(conn, expected_sources: list[str], phrase: str) -> bool:
    if not expected_sources:
        return False
    normalized_phrase = _normalize_text(phrase)
    for source in expected_sources:
        rows = conn.execute(
            """
            SELECT chunks.chunk_text
            FROM chunks
            JOIN documents ON chunks.doc_id = documents.doc_id
            WHERE documents.source_path LIKE ?
            ORDER BY chunks.start_offset
            """,
            (f"%{source}",),
        ).fetchall()
        if not rows:
            continue
        combined = _normalize_text(" ".join(row["chunk_text"] for row in rows))
        if normalized_phrase in combined:
            return True
    return False


def _fetch_chunk_offsets(conn, chunk_id: str) -> tuple[int, int]:
    row = conn.execute(
        "SELECT start_offset, end_offset FROM chunks WHERE chunk_id = ?",
        (chunk_id,),
    ).fetchone()
    assert row is not None
    return int(row["start_offset"]), int(row["end_offset"])


@pytest.mark.slow
def test_golden_retrieval_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = tmp_path / "data"
    monkeypatch.setenv("PSL_DATA_DIR", str(data_dir))

    importlib.reload(config)
    from personal_search_layer.ingestion import pipeline
    from personal_search_layer import indexing, retrieval

    importlib.reload(pipeline)
    importlib.reload(indexing)
    importlib.reload(retrieval)

    summary = pipeline.ingest_path(
        repo_root / "reference_docs" / "smoke_corpus",
        chunk_size=200,
        chunk_overlap=20,
    )
    assert summary.chunks_added > 0

    indexing.build_vector_index(
        model_name=config.MODEL_NAME,
        backend="sentence-transformers",
    )

    cases = _load_cases(repo_root)
    lexical_hits = 0
    hybrid_hits = 0
    with connect(config.DB_PATH) as conn:
        for case in cases:
            query = case["query"]
            expected_sources = case.get("expected_sources", [])
            must_contain = case.get("must_contain", [])
            top_k = int(case.get("top_k", 5))

            lexical = retrieval.search_lexical(query, k=top_k)
            vector = retrieval.search_vector(
                query,
                k=top_k,
                backend="sentence-transformers",
                model_name=config.MODEL_NAME,
            )
            hybrid = retrieval.fuse_hybrid(lexical, vector, k=top_k)

            if expected_sources:
                if _contains_source(lexical.chunks, expected_sources):
                    lexical_hits += 1
                if _contains_source(hybrid.chunks, expected_sources):
                    hybrid_hits += 1
                assert _contains_source(hybrid.chunks, expected_sources)

            for phrase in must_contain:
                assert _phrase_in_sources(conn, expected_sources, phrase)

            for chunk in hybrid.chunks:
                start_offset, end_offset = _fetch_chunk_offsets(conn, chunk.chunk_id)
                assert start_offset < end_offset
                assert (end_offset - start_offset) >= max(1, len(chunk.chunk_text))

    allowed_drop = max(1, int(len(cases) * 0.05))
    assert hybrid_hits + allowed_drop >= lexical_hits
