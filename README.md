# Personal Search Layer

Local-first personal search + agentic RAG that unifies your docs/notes into one query layer. Hybrid retrieval (lexical + embeddings), bounded multi-hop evidence gathering, and claim-by-claim citations with verification/abstain. Includes an evaluation suite + regression tests to track quality over time.

## Mission
Deliver a privacy-preserving, laptop-first search and answer system with traceable evidence. Every answer must be grounded in retrieved chunks, with conflict visibility and clear abstain rationale.

## Non-negotiables
- Local-first by default (no cloud dependencies for core workflows).
- Hybrid retrieval (lexical + vector + fusion) is required.
- Bounded agent loops: max 1 multi-hop expansion + max 1 repair pass.
- Evidence-only prompting with claim-by-claim citations.
- Evaluation-first: golden datasets + regression gates are first-class.

## Architecture (laptop-first)
- **Ingestion**: PDF/text/HTML/DOCX/notebook/CSV/JSON loaders, normalization, metadata extraction, chunking, hashing.
- **Storage**: SQLite for docs/chunks/logs; SQLite FTS5 for lexical index.
- **Vector index**: FAISS for embeddings; metadata in SQLite.
- **Retriever**: hybrid fusion (RRF), intent-aware reranker, filters, caching.
- **Agent engine**: router → retrieve → (optional multi-hop) → synthesize → verify/repair.
- **UI**: Streamlit MVP (search-only + answer mode).
- **Eval/CI**: golden eval sets + adversarial tests + performance profiling.

## Minimal data model
- **documents**: doc_id, source_path, source_type, title, created_at, tags, content_hash
- **chunks**: chunk_id, doc_id, chunk_text, start_offset, end_offset, section, page
- **embeddings**: chunk_id, vector_id, model_name, dim
- **runs/logs**: run_id, query, intent, tool_trace, latency_ms, feedback

## Repo structure
- `src/personal_search_layer/`: core package code
- `scripts/`: CLI entry points (ingest/query)
- `eval/`: evaluation assets and gates
- `data/`: local-only storage (ignored by git)
- `docs/`: detailed architecture + implementation plan

## Setup (Python 3.12 + uv)
The project targets Python 3.12 and uses `uv` for dependency management.

```bash
uv python install 3.12
uv venv --python 3.12
uv sync
```

## Embeddings backend
- Default backend: `sentence-transformers` (downloads model weights on first run).
- Override model: set `PSL_MODEL_NAME` (e.g., `sentence-transformers/all-MiniLM-L6-v2`).
- Pin a specific model revision for reproducible evals: set `PSL_MODEL_REVISION` (HF commit hash or tag).
- Control embedding batch size for indexing speed/memory: set `PSL_EMBED_BATCH_SIZE` (default 64).

## Week 2 routing + eval
- The router assigns a primary intent and recommends pipeline settings (top-k, lexical weight, rerank).
- LOOKUP intents skip vector search by default; rerank runs only for SYNTHESIS/TASK intents.
- Eval reports include metrics (Recall@K, MRR, nDCG), per-intent breakdowns, router accuracy,
  and history snapshots with deltas.

```bash
# Run the eval harness and produce report artifacts
uv run python eval/run_golden_eval.py --top-k 5 --rebuild-index

# Summarize the latest report in a human-readable table
uv run python eval/summarize_eval.py --report-path eval/reports/latest.json
```

## Week 1 usage
All workflows are local-only by default (no cloud calls).

```bash
# Ingest a corpus (adjust path + chunking as needed)
# Data-heavy suffixes are excluded by default; use --include-data to ingest them.
uv run python scripts/ingest.py --path reference_docs/smoke_corpus --chunk-size 1000 --chunk-overlap 120

# Query with hybrid retrieval (rebuilds FAISS index if missing)
uv run python scripts/query.py "smoke corpus keyword" --top-k 8 --rebuild-index

# Run the search-only UI
uv run streamlit run src/personal_search_layer/ui.py

# Database maintenance
uv run python scripts/maintenance.py --integrity-check
uv run python scripts/maintenance.py --vacuum
uv run python scripts/maintenance.py --backup data/backups/search.db

# Reset local data (DB + index)
rm -f data/search.db data/indexes/chunks.faiss
```

## Notes
- Keep personal corpora, indexes, and model weights out of git.
- Update `eval/` whenever retrieval or agent behavior changes.

## Docs
- Architecture: `docs/architecture.md`
- Implementation plan: `docs/implementation-plan.md`
- Code guide: `docs/code-guide.md`
