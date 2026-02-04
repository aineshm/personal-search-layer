"""Vector index build helpers."""

from __future__ import annotations

import time

import faiss

from personal_search_layer.config import (
    DB_PATH,
    EMBEDDING_BACKEND,
    EMBEDDING_DIM,
    FAISS_INDEX_PATH,
    MODEL_NAME,
    ensure_data_dirs,
)
from personal_search_layer.embeddings import embed_texts, get_embedding_dim
from personal_search_layer.models import IndexSummary
from personal_search_layer.storage import (
    clear_embeddings,
    connect,
    get_all_chunks,
    insert_embeddings,
    initialize_schema,
)


def build_vector_index(
    model_name: str = MODEL_NAME,
    dim: int = EMBEDDING_DIM,
    *,
    backend: str = EMBEDDING_BACKEND,
) -> IndexSummary:
    start = time.perf_counter()
    ensure_data_dirs()
    with connect(DB_PATH) as conn:
        initialize_schema(conn)
        rows = get_all_chunks(conn)
        chunk_ids = [row["chunk_id"] for row in rows]
        texts = [row["chunk_text"] for row in rows]
        resolved_dim = get_embedding_dim(backend=backend, model_name=model_name, dim=dim)
        vectors = embed_texts(texts, backend=backend, model_name=model_name, dim=resolved_dim)
        index = faiss.IndexFlatIP(resolved_dim)
        if len(vectors):
            index.add(vectors)
        FAISS_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(FAISS_INDEX_PATH))
        clear_embeddings(conn)
        insert_embeddings(
            conn,
            [(idx, chunk_id, model_name, resolved_dim) for idx, chunk_id in enumerate(chunk_ids)],
        )
        conn.commit()
    elapsed_ms = (time.perf_counter() - start) * 1000
    return IndexSummary(
        chunks_indexed=len(chunk_ids),
        model_name=model_name,
        dim=resolved_dim,
        vectors_written=len(vectors),
        elapsed_ms=elapsed_ms,
    )
