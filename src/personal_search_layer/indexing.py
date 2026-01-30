"""Vector index build helpers (deterministic baseline)."""

from __future__ import annotations

import hashlib
import time
from typing import Iterable

import faiss
import numpy as np

from personal_search_layer.config import (
    DB_PATH,
    EMBEDDING_DIM,
    FAISS_INDEX_PATH,
    MODEL_NAME,
    ensure_data_dirs,
)
from personal_search_layer.models import IndexSummary
from personal_search_layer.storage import (
    clear_embeddings,
    connect,
    get_all_chunks,
    insert_embeddings,
    initialize_schema,
)


def _hash_to_vector(text: str, dim: int) -> np.ndarray:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:4], "little")
    rng = np.random.default_rng(seed)
    vec = rng.normal(size=(dim,)).astype("float32")
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm


def _embed_texts(texts: Iterable[str], dim: int) -> np.ndarray:
    vectors = [_hash_to_vector(text, dim) for text in texts]
    return np.vstack(vectors) if vectors else np.zeros((0, dim), dtype="float32")


def build_vector_index(
    model_name: str = MODEL_NAME, dim: int = EMBEDDING_DIM
) -> IndexSummary:
    start = time.perf_counter()
    ensure_data_dirs()
    with connect(DB_PATH) as conn:
        initialize_schema(conn)
        rows = get_all_chunks(conn)
        chunk_ids = [row["chunk_id"] for row in rows]
        texts = [row["chunk_text"] for row in rows]
        vectors = _embed_texts(texts, dim)
        index = faiss.IndexFlatIP(dim)
        if len(vectors):
            index.add(vectors)
        FAISS_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(FAISS_INDEX_PATH))
        clear_embeddings(conn)
        insert_embeddings(
            conn,
            [
                (idx, chunk_id, model_name, dim)
                for idx, chunk_id in enumerate(chunk_ids)
            ],
        )
        conn.commit()
    elapsed_ms = (time.perf_counter() - start) * 1000
    return IndexSummary(
        chunks_indexed=len(chunk_ids),
        model_name=model_name,
        dim=dim,
        vectors_written=len(vectors),
        elapsed_ms=elapsed_ms,
    )
