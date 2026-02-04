"""Script to ingest data into the personal search layer."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

try:
    from personal_search_layer.config import (
        BLOCKED_SUFFIXES,
        CHUNK_OVERLAP,
        CHUNK_SIZE,
        DATA_DIR,
        MAX_DOC_BYTES,
        MAX_PDF_PAGES,
    )
    from personal_search_layer.ingestion import ingest_path
    from personal_search_layer.telemetry import configure_logging, log_event
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))
    from personal_search_layer.config import (  # type: ignore[reportMissingImports]
        BLOCKED_SUFFIXES,
        CHUNK_OVERLAP,
        CHUNK_SIZE,
        DATA_DIR,
        MAX_DOC_BYTES,
        MAX_PDF_PAGES,
    )
    from personal_search_layer.ingestion import ingest_path  # type: ignore[reportMissingImports]
    from personal_search_layer.telemetry import (  # type: ignore[reportMissingImports]
        configure_logging,
        log_event,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest corpus into the local search layer"
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=DATA_DIR / "corpus",
        help="File or directory to ingest (default: data/corpus)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=CHUNK_SIZE,
        help="Chunk size in characters (default from config)",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=CHUNK_OVERLAP,
        help="Chunk overlap in characters",
    )
    parser.add_argument(
        "--max-doc-bytes",
        type=int,
        default=MAX_DOC_BYTES,
        help="Skip files larger than this size",
    )
    parser.add_argument(
        "--max-pdf-pages",
        type=int,
        default=MAX_PDF_PAGES,
        help="Max PDF pages to ingest per file",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Disable text normalization before chunking/indexing",
    )
    parser.add_argument(
        "--include-data",
        action="store_true",
        help="Include blocked data suffixes (json/csv/tsv/png/zip) during ingestion",
    )
    parser.add_argument(
        "--exclude-suffix",
        action="append",
        default=[],
        help="Additional suffixes to skip (e.g., --exclude-suffix .log)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = configure_logging()
    start = time.perf_counter()
    summary = ingest_path(
        args.path,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        max_doc_bytes=args.max_doc_bytes,
        max_pdf_pages=args.max_pdf_pages,
        normalize=not args.no_normalize,
        exclude_suffixes=_resolve_excluded_suffixes(args),
    )
    elapsed_ms = (time.perf_counter() - start) * 1000
    log_event(
        logger,
        "ingest",
        path=str(args.path),
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        max_doc_bytes=args.max_doc_bytes,
        max_pdf_pages=args.max_pdf_pages,
        normalize=not args.no_normalize,
        exclude_suffixes=sorted(_resolve_excluded_suffixes(args)),
        elapsed_ms=elapsed_ms,
        **summary.to_dict(),
    )
    print("Ingestion summary:", summary.to_dict())


def _resolve_excluded_suffixes(args: argparse.Namespace) -> set[str]:
    base = set() if args.include_data else set(BLOCKED_SUFFIXES)
    extra: set[str] = set()
    for suffix in args.exclude_suffix:
        if not suffix:
            continue
        normalized = suffix.lower()
        extra.add(normalized if normalized.startswith(".") else f".{normalized}")
    return base | extra


if __name__ == "__main__":
    main()
