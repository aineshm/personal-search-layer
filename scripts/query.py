"""Script to query the personal search layer."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys
import time

try:
    from personal_search_layer.config import (
        DB_PATH,
        DEFAULT_TOP_K,
        EMBEDDING_BACKEND,
        EMBEDDING_DIM,
        FAISS_INDEX_PATH,
        MODEL_NAME,
        RRF_K,
    )
    from personal_search_layer.indexing import build_vector_index
    from personal_search_layer.retrieval import (
        fuse_hybrid,
        search_lexical,
        search_vector,
    )
    from personal_search_layer.rerank import rerank_chunks
    from personal_search_layer.router import PrimaryIntent, PipelineSettings, route_query
    from personal_search_layer.storage import connect, initialize_schema, log_run
    from personal_search_layer.telemetry import configure_logging, log_event
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))
    from personal_search_layer.config import (  # type: ignore[reportMissingImports]
        DB_PATH,
        DEFAULT_TOP_K,
        EMBEDDING_BACKEND,
        EMBEDDING_DIM,
        FAISS_INDEX_PATH,
        MODEL_NAME,
        RRF_K,
    )
    from personal_search_layer.indexing import build_vector_index  # type: ignore[reportMissingImports]
    from personal_search_layer.retrieval import (  # type: ignore[reportMissingImports]
        fuse_hybrid,
        search_lexical,
        search_vector,
    )
    from personal_search_layer.rerank import rerank_chunks  # type: ignore[reportMissingImports]
    from personal_search_layer.router import (  # type: ignore[reportMissingImports]
        PrimaryIntent,
        PipelineSettings,
        route_query,
    )
    from personal_search_layer.storage import (  # type: ignore[reportMissingImports]
        connect,
        initialize_schema,
        log_run,
    )
    from personal_search_layer.telemetry import (  # type: ignore[reportMissingImports]
        configure_logging,
        log_event,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query the local search layer")
    parser.add_argument("query", type=str, help="Query string")
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Number of results to return (hybrid). Defaults via router intent.",
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
        "--backend",
        type=str,
        default=EMBEDDING_BACKEND,
        help="Embedding backend (sentence-transformers)",
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=None,
        help="Embedding dimension override (default from config)",
    )
    parser.add_argument(
        "--rrf-k", type=int, default=None, help="Reciprocal rank fusion constant"
    )
    return parser.parse_args()


def maybe_build_index(
    rebuild: bool,
    *,
    model_name: str,
    dim: int | None,
    backend: str,
) -> dict | None:
    if rebuild or not FAISS_INDEX_PATH.exists():
        summary = build_vector_index(
            model_name=model_name, dim=dim or EMBEDDING_DIM, backend=backend
        )
        if summary.chunks_indexed == 0:
            print("Warning: no chunks found; FAISS index contains 0 vectors.")
        print(
            "Index summary:",
            {
                "chunks_indexed": summary.chunks_indexed,
                "vectors_written": summary.vectors_written,
                "model_name": summary.model_name,
                "dim": summary.dim,
                "backend": backend,
                "elapsed_ms": round(summary.elapsed_ms, 2),
            },
        )
        return {
            "chunks_indexed": summary.chunks_indexed,
            "vectors_written": summary.vectors_written,
            "model_name": summary.model_name,
            "dim": summary.dim,
            "backend": backend,
            "elapsed_ms": summary.elapsed_ms,
        }
    return None


def enforce_pipeline_bounds(settings: PipelineSettings) -> PipelineSettings:
    allow_multihop = max(0, min(settings.allow_multihop, 1))
    max_repair_passes = max(0, min(settings.max_repair_passes, 1))
    if allow_multihop == 0:
        max_repair_passes = 0
    if (
        allow_multihop == settings.allow_multihop
        and max_repair_passes == settings.max_repair_passes
    ):
        return settings
    return replace(
        settings,
        allow_multihop=allow_multihop,
        max_repair_passes=max_repair_passes,
    )


def main() -> None:
    args = parse_args()
    logger = configure_logging()
    start = time.perf_counter()
    decision = route_query(args.query)
    default_settings = enforce_pipeline_bounds(decision.recommended_pipeline_settings)
    effective_top_k = args.top_k if args.top_k is not None else default_settings.k
    effective_rrf_k = args.rrf_k if args.rrf_k is not None else RRF_K
    # LOOKUP intents default to lexical-only unless the user explicitly overrides.
    effective_skip_vector = args.skip_vector or decision.primary_intent == PrimaryIntent.LOOKUP
    effective_use_rerank = (
        default_settings.use_rerank
        and decision.primary_intent in {PrimaryIntent.SYNTHESIS, PrimaryIntent.TASK}
    )
    # Rerank is intentionally scoped to synthesis/task queries to keep lookup queries fast.

    index_build = None
    if not effective_skip_vector:
        index_build = maybe_build_index(
            args.rebuild_index,
            model_name=args.model_name,
            dim=args.dim,
            backend=args.backend,
        )

    lexical = search_lexical(args.query, k=effective_top_k)
    vector = (
        search_vector(
            args.query,
            k=effective_top_k,
            backend=args.backend,
            model_name=args.model_name,
            dim=args.dim or EMBEDDING_DIM,
        )
        if not effective_skip_vector
        else None
    )
    hybrid = (
        fuse_hybrid(
            lexical,
            vector,
            k=effective_top_k,
            rrf_k=effective_rrf_k,
            lexical_weight=default_settings.lexical_weight,
        )
        if vector
        else lexical
    )
    if effective_use_rerank:
        hybrid = hybrid.__class__(
            query=hybrid.query,
            mode=hybrid.mode,
            chunks=rerank_chunks(args.query, hybrid.chunks),
            latency_ms=hybrid.latency_ms,
        )
    total_latency_ms = (time.perf_counter() - start) * 1000

    tool_trace = {
        "router": {
            "primary_intent": decision.primary_intent.value,
            "flags": {
                "wants_definition": decision.flags.wants_definition,
                "wants_steps": decision.flags.wants_steps,
                "wants_summary": decision.flags.wants_summary,
            },
            "settings": {
                "k": default_settings.k,
                "lexical_weight": default_settings.lexical_weight,
                "allow_multihop": default_settings.allow_multihop,
                "use_rerank": default_settings.use_rerank,
                "generate_answer": default_settings.generate_answer,
                "verifier_mode": default_settings.verifier_mode.value,
                "max_repair_passes": default_settings.max_repair_passes,
            },
            "signals": decision.signals,
        },
        "lexical": {"k": effective_top_k, "latency_ms": lexical.latency_ms},
        "vector": {
            "k": effective_top_k,
            "latency_ms": vector.latency_ms,
            "backend": args.backend,
            "model_name": args.model_name,
        }
        if vector
        else None,
        "hybrid": {
            "k": effective_top_k,
            "rrf_k": effective_rrf_k,
            "latency_ms": hybrid.latency_ms,
        },
    "rerank": {"enabled": effective_use_rerank},
        "index_build": index_build,
    }

    log_event(
        logger,
        "query",
        query=args.query,
        intent=decision.primary_intent.value,
        top_k=effective_top_k,
        skip_vector=effective_skip_vector,
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
            intent=decision.primary_intent.value,
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
