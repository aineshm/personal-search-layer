"""Script to query the personal search layer."""

from __future__ import annotations

import argparse
import time

from personal_search_layer.config import (
    DB_PATH,
    DEFAULT_TOP_K,
    EMBEDDING_DIM,
    FAISS_INDEX_PATH,
    MODEL_NAME,
    RRF_K,
)
from personal_search_layer.indexing import build_vector_index
from personal_search_layer.retrieval import fuse_hybrid, search_lexical, search_vector
from personal_search_layer.storage import connect, initialize_schema, log_run
from personal_search_layer.telemetry import configure_logging, log_event


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query the local search layer")
    parser.add_argument("query", type=str, help="Query string")
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Number of results to return (hybrid)",
    )
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="Rebuild FAISS index before searching",
    )
    parser.add_argument(
        "--skip-vector", action="store_true", help="Run lexical-only (no vector search)"
    )
    parser.add_argument(
        "--model-name", type=str, default=MODEL_NAME, help="Embedding model name label"
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=None,
        help="Embedding dimension override (default from config)",
    )
    parser.add_argument(
        "--rrf-k", type=int, default=RRF_K, help="Reciprocal rank fusion constant"
    )
    return parser.parse_args()


def maybe_build_index(
    rebuild: bool, *, model_name: str, dim: int | None
) -> dict | None:
    if rebuild or not FAISS_INDEX_PATH.exists():
        summary = build_vector_index(model_name=model_name, dim=dim or EMBEDDING_DIM)
        if summary.chunks_indexed == 0:
            print("Warning: no chunks found; FAISS index contains 0 vectors.")
        print(
            "Index summary:",
            {
                "chunks_indexed": summary.chunks_indexed,
                "vectors_written": summary.vectors_written,
                "model_name": summary.model_name,
                "dim": summary.dim,
                "elapsed_ms": round(summary.elapsed_ms, 2),
            },
        )
        return {
            "chunks_indexed": summary.chunks_indexed,
            "vectors_written": summary.vectors_written,
            "model_name": summary.model_name,
            "dim": summary.dim,
            "elapsed_ms": summary.elapsed_ms,
        }
    return None


def main() -> None:
    args = parse_args()
    logger = configure_logging()
    start = time.perf_counter()

    index_build = None
    if not args.skip_vector:
        index_build = maybe_build_index(
            args.rebuild_index, model_name=args.model_name, dim=args.dim
        )

    lexical = search_lexical(args.query, k=args.top_k)
    vector = search_vector(args.query, k=args.top_k) if not args.skip_vector else None
    hybrid = (
        fuse_hybrid(lexical, vector, k=args.top_k, rrf_k=args.rrf_k)
        if vector
        else lexical
    )
    total_latency_ms = (time.perf_counter() - start) * 1000

    tool_trace = {
        "lexical": {"k": args.top_k, "latency_ms": lexical.latency_ms},
        "vector": {"k": args.top_k, "latency_ms": vector.latency_ms}
        if vector
        else None,
        "hybrid": {
            "k": args.top_k,
            "rrf_k": args.rrf_k,
            "latency_ms": hybrid.latency_ms,
        },
        "index_build": index_build,
    }

    log_event(
        logger,
        "query",
        query=args.query,
        top_k=args.top_k,
        skip_vector=args.skip_vector,
        rebuild_index=args.rebuild_index,
        lexical_latency_ms=lexical.latency_ms,
        vector_latency_ms=vector.latency_ms if vector else None,
        fused_latency_ms=hybrid.latency_ms if hasattr(hybrid, "latency_ms") else None,
        tool_trace=tool_trace,
        total_latency_ms=total_latency_ms,
    )
    with connect(DB_PATH) as conn:
        initialize_schema(conn)
        log_run(
            conn,
            query=args.query,
            intent=None,
            tool_trace=tool_trace,
            latency_ms=total_latency_ms,
        )
        conn.commit()

    results = hybrid.chunks if hasattr(hybrid, "chunks") else []
    for idx, chunk in enumerate(results, start=1):
        print(f"#{idx} {chunk.source_path} score={chunk.score:.4f}")
        print(chunk.chunk_text[:200].strip())


if __name__ == "__main__":
    main()
