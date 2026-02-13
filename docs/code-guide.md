# Code Guide

This is a map of the current codebase and how pieces connect.

## Package layout
`src/personal_search_layer/`
- `config.py`: environment-overridable defaults and paths.
- `embeddings.py`: embedding backends (sentence-transformers).
- `models.py`: shared dataclasses across ingestion/retrieval/orchestration.
- `ingestion/`: loaders, normalization, chunking, pipeline.
- `storage/`: SQLite schema migration/checks + data access helpers + index manifests.
- `indexing.py`: deterministic FAISS build and embedding/manifest mapping.
- `retrieval.py`: lexical, vector, and hybrid retrieval with manifest consistency checks.
- `router.py`: deterministic query intent classification with external policy.
- `router_policy.json`: routing phrases + pipeline defaults (env-overridable via `PSL_ROUTER_POLICY`).
- `rerank.py`: lightweight deterministic reranker.
- `answering.py`: deterministic extractive claims + citation spans.
- `verification.py`: claim support verification, conflict detection, abstain/repair logic.
- `multihop.py`: deterministic follow-up query generation.
- `orchestration.py`: shared search/answer orchestration with strict loop bounds.
- `telemetry.py`: JSON logger for tool traces and metrics.
- `ui.py`: Streamlit search and answer interface.

## CLI entry points
- `scripts/maintenance.py`: schema migration, vacuum, integrity check, backup.
- `scripts/ingest.py`: ingest a corpus; logs summary and metrics.
- `scripts/query.py`: run search/answer flows via orchestration; logs tool traces to DB.

## Key flows
### Migration + Ingestion
1. Run `scripts/maintenance.py --migrate`.
2. `_collect_files()` finds supported files deterministically.
3. `load_document()` returns `LoadedDocument` + `LoadReport`.
4. `_normalize_blocks()` applies normalization if enabled.
5. `chunk_text()` splits blocks into spans with offsets.
6. `insert_document()` and `insert_chunks()` populate SQLite + FTS with deterministic IDs.

### Indexing
1. `get_all_chunks()` loads chunk texts in deterministic order.
2. Sentence-transformers generates embeddings for the FAISS index.
3. FAISS `IndexFlatIP` writes `chunks.faiss`.
4. `insert_embeddings()` stores vector_id -> chunk_id mapping.
5. Active `index_manifest` row is written with snapshot hash + model metadata.

### Retrieval
1. `search_lexical()` builds safe FTS5 query tokens and runs BM25 search.
2. `search_vector()` validates active manifest/path/model/dim/chunk count/snapshot before serving FAISS hits.
3. `fuse_hybrid()` applies RRF with intent-aware lexical weighting.

### Orchestration
1. `route_query()` determines intent and pipeline settings.
2. Retrieval runs (lexical + optional vector + fusion + optional rerank).
3. In answer mode: `synthesize_extractive()` -> `verify_answer()`.
4. If needed and allowed: one multi-hop expansion + one repair pass.
5. Final output is cited answer or abstain rationale with searched queries.

## Eval workflow
- Retrieval eval: `eval/run_golden_eval.py` -> `eval/reports/latest.json`.
- Answer eval: `eval/run_answer_eval.py` -> `eval/reports/answer_latest.json`.
  - Supports isolated eval data prep (`--data-dir`, `--ingest-path`) and hard-gate exits (`--fail-on-hard-gates`).
- Readable retrieval summary: `eval/summarize_eval.py`.

## Commenting guidelines
- Prefer docstrings for module-level behavior.
- Add short comments only where logic is non-obvious.
- Keep tooling deterministic and testable.

## Acceptance criteria (code guide)
- Each module has a clear responsibility and stable boundary.
- Core logic is callable without an LLM.
- Migration, ingestion, retrieval, orchestration, and eval boundaries are explicit.
