"""SQLite schema and data access helpers."""

from __future__ import annotations

import json
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from uuid import uuid4

from personal_search_layer.models import ChunkRecord


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=5.0)
    _configure_connection(conn)
    return conn


def _configure_connection(conn: sqlite3.Connection) -> None:
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute("PRAGMA temp_store = MEMORY")
    conn.execute("PRAGMA busy_timeout = 5000")


def _execute_with_retry(
    conn: sqlite3.Connection,
    sql: str,
    params: tuple | list | None = None,
    *,
    attempts: int = 3,
    base_delay: float = 0.05,
) -> sqlite3.Cursor:
    params = params or ()
    for attempt in range(attempts):
        try:
            return conn.execute(sql, params)
        except sqlite3.OperationalError as exc:
            message = str(exc).lower()
            if "locked" in message or "busy" in message:
                if attempt == attempts - 1:
                    raise
                time.sleep(base_delay * (2**attempt))
                continue
            raise


def _executemany_with_retry(
    conn: sqlite3.Connection,
    sql: str,
    rows: Iterable[tuple],
    *,
    attempts: int = 3,
    base_delay: float = 0.05,
) -> sqlite3.Cursor:
    for attempt in range(attempts):
        try:
            return conn.executemany(sql, rows)
        except sqlite3.OperationalError as exc:
            message = str(exc).lower()
            if "locked" in message or "busy" in message:
                if attempt == attempts - 1:
                    raise
                time.sleep(base_delay * (2**attempt))
                continue
            raise


def initialize_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS documents (
            doc_id TEXT PRIMARY KEY,
            source_path TEXT NOT NULL,
            source_type TEXT NOT NULL,
            title TEXT NOT NULL,
            created_at TEXT NOT NULL,
            tags TEXT,
            content_hash TEXT NOT NULL UNIQUE
        );

        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id TEXT PRIMARY KEY,
            doc_id TEXT NOT NULL,
            chunk_text TEXT NOT NULL,
            start_offset INTEGER NOT NULL,
            end_offset INTEGER NOT NULL,
            section TEXT,
            page INTEGER,
            FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
        );

        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
            chunk_id,
            doc_id,
            chunk_text
        );

        CREATE TABLE IF NOT EXISTS embeddings (
            vector_id INTEGER PRIMARY KEY,
            chunk_id TEXT NOT NULL UNIQUE,
            model_name TEXT NOT NULL,
            dim INTEGER NOT NULL,
            FOREIGN KEY (chunk_id) REFERENCES chunks(chunk_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT PRIMARY KEY,
            query TEXT NOT NULL,
            intent TEXT,
            tool_trace TEXT,
            latency_ms REAL,
            created_at TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id);
        CREATE INDEX IF NOT EXISTS idx_embeddings_chunk_id ON embeddings(chunk_id);
        """
    )


def insert_document(
    conn: sqlite3.Connection,
    *,
    source_path: str,
    source_type: str,
    title: str,
    content_hash: str,
    tags: list[str] | None = None,
) -> tuple[str, bool]:
    row = conn.execute(
        "SELECT doc_id FROM documents WHERE content_hash = ?",
        (content_hash,),
    ).fetchone()
    if row:
        return row["doc_id"], False

    doc_id = str(uuid4())
    _execute_with_retry(
        conn,
        """
        INSERT INTO documents (doc_id, source_path, source_type, title, created_at, tags, content_hash)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            doc_id,
            source_path,
            source_type,
            title,
            datetime.now(timezone.utc).isoformat(),
            json.dumps(tags or []),
            content_hash,
        ),
    )
    return doc_id, True


def insert_chunks(conn: sqlite3.Connection, chunks: Iterable[ChunkRecord]) -> int:
    chunk_rows = [
        (
            chunk.chunk_id,
            chunk.doc_id,
            chunk.chunk_text,
            chunk.start_offset,
            chunk.end_offset,
            chunk.section,
            chunk.page,
        )
        for chunk in chunks
    ]
    if not chunk_rows:
        return 0
    _executemany_with_retry(
        conn,
        """
        INSERT INTO chunks (chunk_id, doc_id, chunk_text, start_offset, end_offset, section, page)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        chunk_rows,
    )
    _executemany_with_retry(
        conn,
        "INSERT INTO chunks_fts (chunk_id, doc_id, chunk_text) VALUES (?, ?, ?)",
        [(row[0], row[1], row[2]) for row in chunk_rows],
    )
    return len(chunk_rows)


def get_all_chunks(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute("SELECT chunk_id, chunk_text FROM chunks").fetchall()


def clear_embeddings(conn: sqlite3.Connection) -> None:
    _execute_with_retry(conn, "DELETE FROM embeddings")


def insert_embeddings(
    conn: sqlite3.Connection,
    rows: Iterable[tuple[int, str, str, int]],
) -> None:
    _executemany_with_retry(
        conn,
        "INSERT INTO embeddings (vector_id, chunk_id, model_name, dim) VALUES (?, ?, ?, ?)",
        rows,
    )


def get_embedding_mapping(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute(
        "SELECT vector_id, chunk_id FROM embeddings ORDER BY vector_id"
    ).fetchall()
    mapping: list[str] = []
    for row in rows:
        vector_id = row["vector_id"]
        if vector_id != len(mapping):
            mapping.extend([""] * (vector_id - len(mapping)))
        mapping.append(row["chunk_id"])
    return mapping


def fetch_chunks_by_ids(
    conn: sqlite3.Connection, chunk_ids: list[str]
) -> list[sqlite3.Row]:
    if not chunk_ids:
        return []
    placeholders = ",".join(["?"] * len(chunk_ids))
    rows = conn.execute(
        f"""
        SELECT chunks.chunk_id, chunks.doc_id, chunks.chunk_text, chunks.page, documents.source_path
        FROM chunks
        JOIN documents ON chunks.doc_id = documents.doc_id
        WHERE chunks.chunk_id IN ({placeholders})
        """,
        chunk_ids,
    ).fetchall()
    by_id = {row["chunk_id"]: row for row in rows}
    return [by_id[chunk_id] for chunk_id in chunk_ids if chunk_id in by_id]


def log_run(
    conn: sqlite3.Connection,
    *,
    query: str,
    intent: str | None,
    tool_trace: dict,
    latency_ms: float,
) -> None:
    _execute_with_retry(
        conn,
        """
        INSERT INTO runs (run_id, query, intent, tool_trace, latency_ms, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            str(uuid4()),
            query,
            intent,
            json.dumps(tool_trace),
            latency_ms,
            datetime.now(timezone.utc).isoformat(),
        ),
    )
