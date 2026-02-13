"""Vector index build helpers."""

from __future__ import annotations

import time
from uuid import uuid4

import faiss

from personal_search_layer.config import (
    DB_PATH,
    EMBEDDING_BACKEND,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_DIM,
    FAISS_INDEX_PATH,
    MODEL_NAME,
    ensure_data_dirs,
)
from personal_search_layer.embeddings import embed_texts, get_embedding_dim
from personal_search_layer.models import IndexSummary
from personal_search_layer.storage import (
    clear_embeddings,
    compute_chunk_snapshot_hash,
    connect,
    deactivate_index_manifests,
    get_all_chunks,
    insert_index_manifest,
    insert_embeddings,
    require_schema,
)
from personal_search_layer.telemetry import configure_logging, log_event


def build_vector_index(
    model_name: str = MODEL_NAME,
    dim: int = EMBEDDING_DIM,
    *,
    backend: str = EMBEDDING_BACKEND,
) -> IndexSummary:
    start = time.perf_counter()
    logger = configure_logging()
    ensure_data_dirs()
    with connect(DB_PATH) as conn:
        require_schema(conn)
        rows = get_all_chunks(conn)
        chunk_ids = [row["chunk_id"] for row in rows]
        texts = [row["chunk_text"] for row in rows]
        snapshot = compute_chunk_snapshot_hash(conn)
        resolved_dim = get_embedding_dim(
            backend=backend, model_name=model_name, dim=dim
        )
        index = faiss.IndexFlatIP(resolved_dim)
        total_chunks = len(texts)
        vectors_written = 0
        if total_chunks:
            batch_size = max(1, EMBEDDING_BATCH_SIZE)
            for batch_start in range(0, total_chunks, batch_size):
                batch_end = min(batch_start + batch_size, total_chunks)
                batch_texts = texts[batch_start:batch_end]
                batch_vectors = embed_texts(
                    batch_texts,
                    backend=backend,
                    model_name=model_name,
                    dim=resolved_dim,
                )
                if len(batch_vectors):
                    index.add(batch_vectors)
                    vectors_written += len(batch_vectors)
                log_event(
                    logger,
                    "index_batch",
                    backend=backend,
                    model_name=model_name,
                    batch_start=batch_start,
                    batch_end=batch_end,
                    total_chunks=total_chunks,
                    vectors_written=vectors_written,
                )
        FAISS_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(FAISS_INDEX_PATH))
        clear_embeddings(conn)
        insert_embeddings(
            conn,
            [
                (idx, chunk_id, model_name, resolved_dim)
                for idx, chunk_id in enumerate(chunk_ids)
            ],
        )
        deactivate_index_manifests(conn)
        insert_index_manifest(
            conn,
            index_id=f"idx_{uuid4()}",
            model_name=model_name,
            dim=resolved_dim,
            chunk_count=len(chunk_ids),
            chunk_snapshot_hash=snapshot,
            faiss_path=str(FAISS_INDEX_PATH),
            active=1,
        )
        conn.commit()
    elapsed_ms = (time.perf_counter() - start) * 1000
    log_event(
        logger,
        "index_complete",
        backend=backend,
        model_name=model_name,
        dim=resolved_dim,
        chunks_indexed=len(chunk_ids),
        vectors_written=vectors_written,
        elapsed_ms=elapsed_ms,
    )
    return IndexSummary(
        chunks_indexed=len(chunk_ids),
        model_name=model_name,
        dim=resolved_dim,
        vectors_written=vectors_written,
        elapsed_ms=elapsed_ms,
    )
