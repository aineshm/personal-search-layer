"""Chunking utilities."""

from __future__ import annotations

from personal_search_layer.models import ChunkSpan, TextBlock


def chunk_text(
    blocks: list[TextBlock],
    *,
    chunk_size: int = 1000,
    overlap: int = 120,
) -> list[ChunkSpan]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    spans: list[ChunkSpan] = []
    cursor = 0
    for block in blocks:
        text = block.text.strip()
        if not text:
            continue
        block_cursor = 0
        while block_cursor < len(text):
            end = min(block_cursor + chunk_size, len(text))
            chunk_text = text[block_cursor:end]
            spans.append(
                ChunkSpan(
                    text=chunk_text,
                    start_offset=cursor + block_cursor,
                    end_offset=cursor + end,
                    page=block.page,
                    section=block.section,
                )
            )
            if end == len(text):
                break
            block_cursor = end - overlap
        cursor += len(text)
    return spans
