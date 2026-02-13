"""Script to query the personal search layer."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

try:
    from personal_search_layer.config import (
        DB_PATH,
        EMBEDDING_BACKEND,
        EMBEDDING_DIM,
        FAISS_INDEX_PATH,
        MODEL_NAME,
    )
    from personal_search_layer.indexing import build_vector_index
    from personal_search_layer.orchestration import run_query
    from personal_search_layer.storage import connect, log_run, require_schema
    from personal_search_layer.telemetry import configure_logging, log_event
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))
    from personal_search_layer.config import (  # type: ignore[reportMissingImports]
        DB_PATH,
        EMBEDDING_BACKEND,
        EMBEDDING_DIM,
        FAISS_INDEX_PATH,
        MODEL_NAME,
    )
    from personal_search_layer.indexing import build_vector_index  # type: ignore[reportMissingImports]
    from personal_search_layer.orchestration import run_query  # type: ignore[reportMissingImports]
    from personal_search_layer.storage import (  # type: ignore[reportMissingImports]
        connect,
        log_run,
        require_schema,
    )
    from personal_search_layer.telemetry import (  # type: ignore[reportMissingImports]
        configure_logging,
        log_event,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query the local search layer")
    parser.add_argument("query", type=str, help="Query string")
    parser.add_argument(
        "--mode",
        choices=["search", "answer"],
        default="search",
        help="Run retrieval-only search mode or verified answer mode",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Number of results to return (defaults via router intent).",
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
        payload = {
            "chunks_indexed": summary.chunks_indexed,
            "vectors_written": summary.vectors_written,
            "model_name": summary.model_name,
            "dim": summary.dim,
            "backend": backend,
            "elapsed_ms": round(summary.elapsed_ms, 2),
        }
        print("Index summary:", payload)
        return payload
    return None


def _print_search_results(result) -> None:
    for idx, chunk in enumerate(result.chunks, start=1):
        print(f"#{idx} {chunk.source_path} score={chunk.score:.4f}")
        print(chunk.chunk_text[:200].strip())


def _print_answer_results(result) -> None:
    draft = result.draft_answer
    verification = result.verification
    print(
        f"intent={result.intent} mode={result.mode} latency_ms={result.latency_ms:.2f}"
    )
    if not draft:
        print("No draft answer generated.")
        return

    if verification and verification.abstain:
        print("ABSTAINED")
        print(f"Reason: {verification.abstain_reason or 'insufficient evidence'}")
        searched = ", ".join(verification.searched_queries or draft.searched_queries)
        print(f"Searched queries: {searched}")
        if verification.conflicts:
            print("Conflicts:")
            for conflict in verification.conflicts:
                print(f"- {conflict}")
        return

    print("Answer:")
    print(draft.answer_text)
    print("Claims and citations:")
    for claim in draft.claims:
        print(f"- {claim.claim_id}: {claim.text}")
        for citation in claim.citations:
            span = f"{citation.quote_span_start}:{citation.quote_span_end}"
            page = citation.page if citation.page is not None else "n/a"
            print(
                "  citation "
                f"chunk={citation.chunk_id} source={citation.source_path} page={page} span={span}"
            )
    if verification and verification.conflicts:
        print("Conflicts:")
        for conflict in verification.conflicts:
            print(f"- {conflict}")


def main() -> None:
    args = parse_args()
    logger = configure_logging()
    start = time.perf_counter()

    index_build = None
    if not args.skip_vector:
        index_build = maybe_build_index(
            args.rebuild_index,
            model_name=args.model_name,
            dim=args.dim,
            backend=args.backend,
        )

    result = run_query(
        args.query,
        mode=args.mode,
        top_k=args.top_k,
        skip_vector=args.skip_vector,
    )

    total_latency_ms = (time.perf_counter() - start) * 1000
    result.tool_trace["index_build"] = index_build

    log_event(
        logger,
        "query",
        query=args.query,
        mode=args.mode,
        intent=result.intent,
        top_k=args.top_k,
        skip_vector=args.skip_vector,
        rebuild_index=args.rebuild_index,
        total_latency_ms=total_latency_ms,
        tool_trace=result.tool_trace,
    )
    with connect(DB_PATH) as conn:
        require_schema(conn)
        log_run(
            conn,
            query=args.query,
            intent=result.intent,
            tool_trace=result.tool_trace,
            latency_ms=total_latency_ms,
        )
        conn.commit()

    if args.mode == "search":
        _print_search_results(result)
    else:
        _print_answer_results(result)


if __name__ == "__main__":
    main()
