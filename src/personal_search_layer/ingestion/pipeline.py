"""Ingestion pipeline for documents and chunking."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from personal_search_layer.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DB_PATH,
    MAX_DOC_BYTES,
    MAX_PDF_PAGES,
    NORMALIZE_TEXT,
    ensure_data_dirs,
)
from personal_search_layer.ingestion.chunking import chunk_text
from personal_search_layer.ingestion.loaders import SUPPORTED_SUFFIXES, load_document
from personal_search_layer.ingestion.normalization import normalize_text
from personal_search_layer.models import ChunkRecord, IngestSummary, TextBlock
from personal_search_layer.storage import (
    connect,
    initialize_schema,
    insert_chunks,
    insert_document,
)


def ingest_path(
    path: Path,
    *,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    max_doc_bytes: int = MAX_DOC_BYTES,
    max_pdf_pages: int = MAX_PDF_PAGES,
    normalize: bool = NORMALIZE_TEXT,
) -> IngestSummary:
    ensure_data_dirs()
    files = _collect_files(path)
    summary = IngestSummary(
        files_seen=len(files),
        documents_added=0,
        chunks_added=0,
        duplicates_skipped=0,
        files_skipped=0,
        skip_reasons={},
        pages_skipped_empty=0,
        pages_skipped_limit=0,
    )
    with connect(DB_PATH) as conn:
        initialize_schema(conn)
        for file_path in files:
            doc, report = load_document(
                file_path,
                max_doc_bytes=max_doc_bytes,
                max_pdf_pages=max_pdf_pages,
            )
            summary.pages_skipped_empty += report.pages_skipped_empty
            summary.pages_skipped_limit += report.pages_skipped_limit
            if report.skip_reason:
                summary.files_skipped += 1
                summary.skip_reasons[report.skip_reason] = (
                    summary.skip_reasons.get(report.skip_reason, 0) + 1
                )
                continue
            if doc is None:
                summary.files_skipped += 1
                summary.skip_reasons["load_failed"] = (
                    summary.skip_reasons.get("load_failed", 0) + 1
                )
                continue
            blocks = _normalize_blocks(doc.blocks, normalize=normalize)
            if not blocks:
                summary.files_skipped += 1
                summary.skip_reasons["empty_after_normalization"] = (
                    summary.skip_reasons.get("empty_after_normalization", 0) + 1
                )
                continue
            doc_id, inserted = insert_document(
                conn,
                source_path=doc.source_path,
                source_type=doc.source_type,
                title=doc.title,
                content_hash=doc.content_hash,
            )
            if not inserted:
                summary.duplicates_skipped += 1
                continue
            summary.documents_added += 1
            spans = chunk_text(blocks, chunk_size=chunk_size, overlap=chunk_overlap)
            chunk_records = [
                ChunkRecord(
                    chunk_id=str(uuid4()),
                    doc_id=doc_id,
                    chunk_text=span.text,
                    start_offset=span.start_offset,
                    end_offset=span.end_offset,
                    section=span.section,
                    page=span.page,
                )
                for span in spans
            ]
            summary.chunks_added += insert_chunks(conn, chunk_records)
        conn.commit()
    return summary


def _normalize_blocks(blocks: list[TextBlock], *, normalize: bool) -> list[TextBlock]:
    if not normalize:
        return [block for block in blocks if block.text.strip()]
    normalized: list[TextBlock] = []
    for block in blocks:
        text = normalize_text(block.text)
        if not text:
            continue
        normalized.append(TextBlock(text=text, page=block.page, section=block.section))
    return normalized


def _collect_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path] if path.suffix.lower() in SUPPORTED_SUFFIXES else []
    files: list[Path] = []
    for candidate in path.rglob("*"):
        if candidate.is_file() and candidate.suffix.lower() in SUPPORTED_SUFFIXES:
            files.append(candidate)
    return files
