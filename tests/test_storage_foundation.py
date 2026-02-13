from pathlib import Path

from personal_search_layer.models import ChunkRecord
from personal_search_layer.storage import (
    connect,
    get_all_chunks,
    initialize_schema,
    insert_chunks,
    insert_document,
    require_schema,
)


def test_insert_document_uses_stable_doc_id(tmp_path: Path) -> None:
    db_path = tmp_path / "search.db"
    with connect(db_path) as conn:
        initialize_schema(conn)
        doc_id, inserted = insert_document(
            conn,
            source_path="/tmp/file.txt",
            source_type="text",
            title="file",
            content_hash="abcd" * 16,
        )
        assert inserted is True
        assert doc_id.startswith("doc_")

        same_doc_id, inserted_again = insert_document(
            conn,
            source_path="/tmp/file.txt",
            source_type="text",
            title="file",
            content_hash="abcd" * 16,
        )
        assert inserted_again is False
        assert same_doc_id == doc_id


def test_get_all_chunks_is_deterministic_order(tmp_path: Path) -> None:
    db_path = tmp_path / "search.db"
    with connect(db_path) as conn:
        initialize_schema(conn)
        doc_id, _ = insert_document(
            conn,
            source_path="/tmp/file.txt",
            source_type="text",
            title="file",
            content_hash="ef01" * 16,
        )
        insert_chunks(
            conn,
            [
                ChunkRecord("chunk_b", doc_id, "b", 0, 1, None, None),
                ChunkRecord("chunk_a", doc_id, "a", 2, 3, None, None),
            ],
        )
        conn.commit()

        rows = get_all_chunks(conn)
        assert [row["chunk_id"] for row in rows] == ["chunk_a", "chunk_b"]


def test_require_schema_fails_before_migration(tmp_path: Path) -> None:
    db_path = tmp_path / "search.db"
    with connect(db_path) as conn:
        try:
            require_schema(conn)
            assert False, "expected require_schema to fail before migration"
        except RuntimeError:
            pass

        initialize_schema(conn)
        require_schema(conn)
