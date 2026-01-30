# Code Guide

This is a map of the current codebase and how pieces connect.

## Package layout
`src/personal_search_layer/`
- `config.py`: Environment-overridable defaults and paths.
- `models.py`: Shared dataclasses used across ingestion and retrieval.
- `ingestion/`: Loaders, normalization, chunking, pipeline.
- `storage/`: SQLite schema + data access helpers.
- `indexing.py`: Build deterministic FAISS index and embeddings mapping.
- `retrieval.py`: Lexical, vector, and hybrid retrieval.
- `telemetry.py`: JSON logger for tool traces and metrics.
- `ui.py`: Streamlit search-only interface.

## CLI entry points
- `scripts/ingest.py`: Ingest a corpus; logs summary and metrics.
- `scripts/query.py`: Query lexical/vector/hybrid; logs tool trace to DB.
- `scripts/maintenance.py`: Vacuum, integrity check, backup.

## Key flows
### Ingestion
1. `_collect_files()` finds supported files.
2. `load_document()` returns `LoadedDocument` + `LoadReport`.
3. `_normalize_blocks()` applies normalization if enabled.
4. `chunk_text()` splits blocks into `ChunkSpan`s with offsets.
5. `insert_document()` and `insert_chunks()` populate SQLite + FTS.

### Indexing
1. `get_all_chunks()` loads chunk texts.
2. `_hash_to_vector()` generates deterministic embeddings.
3. FAISS `IndexFlatIP` writes `chunks.faiss`.
4. `insert_embeddings()` stores vector_id -> chunk_id mapping.

### Retrieval
1. `search_lexical()` queries FTS5 BM25.
2. `search_vector()` queries FAISS + mapping.
3. `fuse_hybrid()` applies RRF to merge rankings.

## Extension points
- Add reranker to `retrieval.py` after fusion.
- Add router/multi-hop agents as orchestration layer.
- Swap embeddings for a local model and persist vectors for reuse.
- Add verification/abstain logic in a new module, keeping tools pure.

## Commenting guidelines
- Prefer docstrings for module-level behavior.
- Add short comments only where logic is non-obvious or subtle.
- Keep tooling deterministic and testable.

## Acceptance criteria (code guide)
- Each module has a clear responsibility and public surface area.
- No function requires the LLM to execute core logic.
- Retrieval, ingestion, and storage boundaries are explicit and stable.
