# Architecture

## Overview
Personal Search Layer is a local-first retrieval and evidence system. The current implementation supports:
- Ingestion of text, HTML, PDF, DOCX, notebook, CSV/TSV, and JSON sources (data formats can be excluded by default).
- Normalization and chunking.
- SQLite storage with FTS5 lexical index.
- FAISS vector index (sentence-transformers by default).
- Hybrid retrieval via Reciprocal Rank Fusion (RRF).
- Search-only UI and CLI workflows.

## High-level data flow
1. Ingest files from a path.
2. Load and normalize text blocks.
3. Chunk blocks into spans with metadata (page/section).
4. Store documents and chunks in SQLite + FTS5.
5. Build a FAISS index over chunks.
6. Route query to an intent + pipeline settings.
7. Query: lexical + vector search -> hybrid fusion -> optional rerank.
8. UI/CLI returns evidence with metadata and logs run traces.

## Component map
- Ingestion: `src/personal_search_layer/ingestion/`
- `loaders.py`: load PDF/HTML/DOCX/notebook/CSV/JSON/text into `TextBlock`s.
  - `normalization.py`: NFKC + lowercase + whitespace collapse.
  - `chunking.py`: produce `ChunkSpan` with offsets and metadata.
  - `pipeline.py`: end-to-end ingestion, dedupe by content hash.
- Storage: `src/personal_search_layer/storage/`
  - SQLite schema, FTS5, embedding metadata mapping, run logging.
- Indexing: `src/personal_search_layer/indexing.py`
  - Local embedding model (sentence-transformers); FAISS index build.
- Retrieval: `src/personal_search_layer/retrieval.py`
  - Lexical search (FTS5 BM25), vector search (FAISS), RRF fusion.
- Routing: `src/personal_search_layer/router.py`
  - Deterministic primary intent classification + pipeline settings.
- Rerank: `src/personal_search_layer/rerank.py`
  - Lightweight reranker stub (intent-gated in the query pipeline).
- UI: `src/personal_search_layer/ui.py`
  - Streamlit search-only experience with evidence view.
- Telemetry: `src/personal_search_layer/telemetry.py`
  - Structured JSON logging for ingest/query/UI.
- CLI: `scripts/ingest.py`, `scripts/query.py`, `scripts/maintenance.py`

## Storage schema (SQLite)
Tables:
- `documents`: doc_id, source_path, source_type, title, created_at, tags, content_hash
- `chunks`: chunk_id, doc_id, chunk_text, start_offset, end_offset, section, page
- `chunks_fts`: FTS5 virtual table for `chunk_text`
- `embeddings`: vector_id, chunk_id, model_name, dim
- `runs`: run_id, query, intent, tool_trace, latency_ms, created_at

## Retrieval semantics
- Lexical search: FTS5 BM25; negative scores are inverted to positive.
- Vector search: FAISS IndexFlatIP over normalized embedding vectors.
- Fusion: RRF with configurable `rrf_k` and intent-aware lexical weighting.
- Rerank: optional lexical-overlap reranker, enabled only for SYNTHESIS/TASK.

## Non-negotiables mapped to architecture
- Local-first: SQLite + FAISS, no hosted dependencies in default workflows.
- Hybrid retrieval: lexical + vector + fusion.
- Bounded loops: limit multi-hop and repair in orchestration layer (Week 2+).
- Evidence-only: retrieval returns chunks with metadata for citations.
- Evaluation-first: smoke corpus + tests; expand to eval suite (Week 2+).

## Evaluation artifacts
- `eval/run_golden_eval.py` produces `eval/reports/latest.json` plus timestamped history.
- Reports include metric deltas versus the previous run and router accuracy.

## Extension points
- Swap embedding models as needed for local workflows.
- Add reranker after fusion.
- Add router and multi-hop retrieval agents.
- Add verifier/repair for claim-by-claim citation checking.
- Add export tools (markdown/pdf) for answer mode.

## Acceptance criteria (architecture)
- Ingestion path produces `documents` and `chunks` rows with page/section metadata where available.
- FTS5 index exists and returns lexical matches for simple queries.
- FAISS index exists and returns vector matches for simple queries.
- Hybrid fusion returns stable top-k results across runs.
- Logs include tool traces and latency metrics for ingest/query/UI.
