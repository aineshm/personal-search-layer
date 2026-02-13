"""SQLite storage helpers."""

from .db import (
    clear_embeddings,
    compute_chunk_snapshot_hash,
    connect,
    deactivate_index_manifests,
    fetch_chunks_by_ids,
    get_all_chunks,
    get_active_index_manifest,
    get_embedding_mapping,
    initialize_schema,
    insert_index_manifest,
    insert_chunks,
    insert_document,
    insert_embeddings,
    migrate_schema,
    require_schema,
    log_run,
)

__all__ = [
    "clear_embeddings",
    "compute_chunk_snapshot_hash",
    "connect",
    "deactivate_index_manifests",
    "fetch_chunks_by_ids",
    "get_all_chunks",
    "get_active_index_manifest",
    "get_embedding_mapping",
    "initialize_schema",
    "insert_index_manifest",
    "insert_chunks",
    "insert_document",
    "insert_embeddings",
    "migrate_schema",
    "require_schema",
    "log_run",
]
