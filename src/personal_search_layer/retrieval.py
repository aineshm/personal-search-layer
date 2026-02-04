"""Retrieval utilities: lexical (FTS5), vector (FAISS), and hybrid fusion."""

from __future__ import annotations

import time
from collections import defaultdict

import faiss
import numpy as np

from personal_search_layer.config import (
    DB_PATH,
    EMBEDDING_BACKEND,
    EMBEDDING_DIM,
    FAISS_INDEX_PATH,
    MODEL_NAME,
    RRF_K,
)
from personal_search_layer.embeddings import embed_query, get_embedding_dim
from personal_search_layer.models import ScoredChunk, SearchResult
from personal_search_layer.storage import (
    connect,
    fetch_chunks_by_ids,
    get_embedding_mapping,
    initialize_schema,
)


def search_lexical(query: str, k: int = 8) -> SearchResult:
    start = time.perf_counter()
    with connect(DB_PATH) as conn:
        initialize_schema(conn)
        rows = conn.execute(
            """
            SELECT chunk_id, bm25(chunks_fts) AS score
            FROM chunks_fts
            WHERE chunks_fts MATCH ?
            ORDER BY score LIMIT ?
            """,
            (query, k),
        ).fetchall()
        chunk_ids = [row["chunk_id"] for row in rows]
        chunk_rows = fetch_chunks_by_ids(conn, chunk_ids)
    scored: list[ScoredChunk] = []
    for row, chunk in zip(rows, chunk_rows, strict=False):
        scored.append(
            ScoredChunk(
                chunk_id=chunk["chunk_id"],
                doc_id=chunk["doc_id"],
                score=float(-row["score"]),
                chunk_text=chunk["chunk_text"],
                source_path=chunk["source_path"],
                page=chunk["page"],
            )
        )
    latency_ms = (time.perf_counter() - start) * 1000
    return SearchResult(
        query=query, mode="lexical", chunks=scored, latency_ms=latency_ms
    )


def _filter_faiss_hits(
    indices: np.ndarray, scores: np.ndarray, mapping: list[str]
) -> list[tuple[float, str]]:
    """Filter FAISS search outputs to valid chunk ids in order."""
    hits: list[tuple[float, str]] = []
    if not mapping:
        return hits
    for idx, score in zip(indices.tolist(), scores.tolist(), strict=False):
        if idx < 0 or idx >= len(mapping):
            continue
        chunk_id = mapping[idx]
        if not chunk_id:
            continue
        hits.append((float(score), chunk_id))
    return hits


def search_vector(
    query: str,
    k: int = 8,
    dim: int = EMBEDDING_DIM,
    *,
    backend: str = EMBEDDING_BACKEND,
    model_name: str = MODEL_NAME,
) -> SearchResult:
    start = time.perf_counter()
    if not FAISS_INDEX_PATH.exists():
        return SearchResult(query=query, mode="vector", chunks=[], latency_ms=0.0)
    index = faiss.read_index(str(FAISS_INDEX_PATH))
    resolved_dim = get_embedding_dim(backend=backend, model_name=model_name, dim=dim)
    query_vec = embed_query(query, backend=backend, model_name=model_name, dim=resolved_dim)
    scores, indices = index.search(np.asarray([query_vec]), k)
    with connect(DB_PATH) as conn:
        initialize_schema(conn)
        mapping = get_embedding_mapping(conn)
        hits = _filter_faiss_hits(indices[0], scores[0], mapping)
        chunk_ids = [chunk_id for _, chunk_id in hits]
        chunk_rows = fetch_chunks_by_ids(conn, chunk_ids)
    scored: list[ScoredChunk] = []
    for (score, _), chunk in zip(hits, chunk_rows, strict=False):
        scored.append(
            ScoredChunk(
                chunk_id=chunk["chunk_id"],
                doc_id=chunk["doc_id"],
                score=float(score),
                chunk_text=chunk["chunk_text"],
                source_path=chunk["source_path"],
                page=chunk["page"],
            )
        )
    latency_ms = (time.perf_counter() - start) * 1000
    return SearchResult(
        query=query, mode="vector", chunks=scored, latency_ms=latency_ms
    )


def fuse_hybrid(
    lexical: SearchResult,
    vector: SearchResult,
    k: int = 8,
    rrf_k: int = RRF_K,
) -> SearchResult:
    start = time.perf_counter()
    scores: dict[str, float] = defaultdict(float)
    lookup: dict[str, ScoredChunk] = {}
    for rank, chunk in enumerate(lexical.chunks, start=1):
        scores[chunk.chunk_id] += 1.0 / (rrf_k + rank)
        lookup[chunk.chunk_id] = chunk
    for rank, chunk in enumerate(vector.chunks, start=1):
        scores[chunk.chunk_id] += 1.0 / (rrf_k + rank)
        lookup.setdefault(chunk.chunk_id, chunk)
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:k]
    fused = [
        ScoredChunk(
            chunk_id=chunk_id,
            doc_id=lookup[chunk_id].doc_id,
            score=score,
            chunk_text=lookup[chunk_id].chunk_text,
            source_path=lookup[chunk_id].source_path,
            page=lookup[chunk_id].page,
        )
        for chunk_id, score in ranked
    ]
    latency_ms = (time.perf_counter() - start) * 1000
    return SearchResult(
        query=lexical.query, mode="hybrid", chunks=fused, latency_ms=latency_ms
    )
