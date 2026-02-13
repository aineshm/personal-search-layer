# Architecture

## Overview
Personal Search Layer is a local-first retrieval and evidence system. The current implementation supports:
- Ingestion of text, HTML, PDF, DOCX, notebook, CSV/TSV, and JSON sources (data formats can be excluded by default).
- Normalization and deterministic chunking.
- SQLite storage with FTS5 lexical index.
- FAISS vector index (sentence-transformers by default).
- Hybrid retrieval via Reciprocal Rank Fusion (RRF).
- Deterministic answer orchestration with bounded loops, claim citations, verification, and abstain/conflict handling.

## High-level data flow
1. Run schema migration (`scripts/maintenance.py --migrate`).
2. Ingest files from a path.
3. Load and normalize text blocks.
4. Chunk blocks into spans with metadata (page/section).
5. Store documents/chunks in SQLite + FTS5.
6. Build a FAISS index over chunks and write an active index manifest.
7. Route query to intent + pipeline settings.
8. Search mode: lexical + vector -> hybrid fusion -> optional rerank.
9. Answer mode: synthesize extractive claims -> verify -> optional one-hop expansion -> optional one repair -> final answer or abstain.

## Component map
- Ingestion: `src/personal_search_layer/ingestion/`
  - `loaders.py`: load PDF/HTML/DOCX/notebook/CSV/JSON/text into `TextBlock`s.
  - `normalization.py`: NFKC + lowercase + whitespace collapse.
  - `chunking.py`: produce `ChunkSpan` with offsets and metadata.
  - `pipeline.py`: end-to-end ingestion, dedupe by content hash, deterministic chunk IDs.
- Storage: `src/personal_search_layer/storage/`
  - SQLite schema, migrations, strict schema checks, index manifests, embedding mapping, run logging.
- Indexing: `src/personal_search_layer/indexing.py`
  - Local embedding model (sentence-transformers); FAISS index build + manifest write.
- Retrieval: `src/personal_search_layer/retrieval.py`
  - Lexical search (FTS5 BM25), vector search (FAISS), RRF fusion, manifest consistency checks.
- Routing: `src/personal_search_layer/router.py`
  - Deterministic intent classification with externalized policy (`router_policy.json`).
- Answering: `src/personal_search_layer/answering.py`
  - Deterministic extractive claim synthesis + citation spans.
- Verification: `src/personal_search_layer/verification.py`
  - Claim support checks, conflict detection, abstain decisions, deterministic repair.
- Multi-hop: `src/personal_search_layer/multihop.py`
  - Deterministic follow-up query proposal (bounded to 1 hop).
- Orchestration: `src/personal_search_layer/orchestration.py`
  - Shared query pipeline for search/answer modes with strict loop bounds.
- UI: `src/personal_search_layer/ui.py`
  - Streamlit search and answer modes.
- Telemetry: `src/personal_search_layer/telemetry.py`
  - Structured JSON logging for ingest/query/UI.
- CLI: `scripts/ingest.py`, `scripts/query.py`, `scripts/maintenance.py`

## Storage schema (SQLite)
Tables:
- `schema_meta`: schema version metadata
- `documents`: doc_id, source_path, source_type, title, created_at, tags, content_hash
- `chunks`: chunk_id, doc_id, chunk_text, start_offset, end_offset, section, page
- `chunks_fts`: FTS5 virtual table for `chunk_text`
- `embeddings`: vector_id, chunk_id, model_name, dim
- `index_manifests`: index_id, model_name, dim, chunk_count, chunk_snapshot_hash, faiss_path, created_at, active
- `runs`: run_id, query, intent, tool_trace, latency_ms, created_at

## Retrieval semantics
- Lexical search: FTS5 BM25 with deterministic tokenized query building.
- Vector search: FAISS `IndexFlatIP` over normalized embedding vectors.
- Vector safety checks: active index manifest must match FAISS path/model/dim/chunk count/chunk snapshot hash.
- Fusion: RRF with configurable `rrf_k` and intent-aware lexical weighting.
- Rerank: optional lexical-overlap reranker, enabled for synthesis/task-style intents.

## Non-negotiables mapped to architecture
- Local-first: SQLite + FAISS, no hosted dependencies in default workflows.
- Hybrid retrieval: lexical + vector + fusion.
- Bounded loops: max 1 multi-hop expansion + max 1 repair pass.
- Evidence-only: answer mode emits claim-level citations and abstains on unsupported evidence.
- Evaluation-first: retrieval + answer eval suites with report artifacts and deltas.

## Evaluation artifacts
- `eval/run_golden_eval.py` produces `eval/reports/latest.json` plus timestamped history.
- `eval/run_answer_eval.py` produces `eval/reports/answer_latest.json` plus timestamped history.
- Reports include metric deltas versus previous snapshots when available.

## Acceptance criteria (architecture)
- Schema migration is explicit and required before read/query flows.
- Ingestion path produces `documents` and `chunks` rows with metadata where available.
- FTS5 and FAISS indexes both return expected matches on smoke corpus checks.
- Hybrid fusion returns stable top-k results across repeated runs.
- Answer mode returns either fully cited claims or abstain with searched-query rationale.
- Logs include tool traces and latency metrics for ingest/query/UI.
