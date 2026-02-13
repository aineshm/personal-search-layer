# Project Status Summary

_Last updated: 2026-02-13_

## What’s completed (and how)

### Environment setup
- Created a UV-managed Python 3.12 virtual environment at `.venv` and synced dependencies with `uv sync`.
- Installed dev dependencies (`pytest`, `ruff`) via `uv sync --extra dev`.

### Week 1 acceptance checks
- **Ingestion:** Smoke corpus successfully ingested via `scripts/ingest.py` (non-zero chunks).
- **Indexing:** FAISS vector index built from sentence-transformers embeddings via `build_vector_index`.
- **Query:** `scripts/query.py` returns hits for “smoke corpus keyword”.
- **Tests:** `pytest -q` passes after updates.

### Retrieval quality gates (golden retrieval)
- Added a **golden retrieval case set** in `eval/golden_retrieval.jsonl` with comprehensive queries covering all smoke corpus documents.
- Added a **golden evaluation runner** `eval/run_golden_eval.py` to compute Recall@K, MRR, and nDCG for lexical, vector, and hybrid modes.
- Eval runs now emit a **report artifact** at `eval/reports/latest.json` with model metadata and git commit hash.
- Added a **golden retrieval test** `tests/test_golden_retrieval.py` that:
  - Ingests the smoke corpus into a temp data dir.
  - Builds a **sentence-transformers** vector index (no hash fallback).
  - Validates expected sources appear in **hybrid** top‑K.
  - Checks phrase presence, score ordering, and chunk boundary metadata (offset sanity).
  - Asserts hybrid recall ≥ lexical recall.

### Vector retrieval only (no hash backend usage)
- Removed hash-backend references from documentation.
- Updated embedding unit tests to exercise the sentence-transformers path (via a deterministic dummy model for unit tests).
- Golden retrieval tests and evals now use **real vector embeddings** in the pipeline.
- Optional **model revision pinning** via `PSL_MODEL_REVISION` for reproducible evals.

### Week 2 delivery (completed)
- Implemented **primary-intent router** with pipeline settings (k, lexical weight, rerank, multihop).
- Enforced intent-aware pipeline defaults in `scripts/query.py`:
  - LOOKUP skips vector search by default.
  - Rerank runs only for SYNTHESIS/TASK.
- Added a **reranker stub** and wired it into the query pipeline.
- Expanded golden retrieval cases with paraphrases to stress vector retrieval.
- Expanded eval harness with Recall/MRR/nDCG, per-intent metrics, router accuracy, and report artifacts.
- Added **history snapshots + metric deltas** in `eval/reports/`.
- Added a **human-readable eval summary helper** `eval/summarize_eval.py` and tests.

## Key files added/updated
- `eval/golden_retrieval.jsonl`: Expanded golden cases for smoke corpus.
- `eval/run_golden_eval.py`: Local eval runner with report artifacts + history + deltas.
- `eval/summarize_eval.py`: Human-readable report summary helper.
- `tests/test_golden_retrieval.py`: Comprehensive regression test for retrieval quality + metadata sanity.
- `tests/test_embeddings.py`: Unit tests now aligned with sentence-transformers path.
- `docs/architecture.md`, `docs/code-guide.md`, `docs/implementation-plan.md`, `README.md`: Updated to reflect vector-only embeddings.
- `pytest.ini`: Added `slow` marker for long-running retrieval tests.

## How to validate (quick checklist)
- Run fast unit tests (excludes slow golden tests): `pytest -q`
- Run golden retrieval gate: `pytest -q -m slow`
- Run eval report: `uv run python eval/run_golden_eval.py --top-k 5 --rebuild-index`
- Summarize eval report: `uv run python eval/summarize_eval.py --report-path eval/reports/latest.json`
- Run smoke ingest/query:
  - `uv run python scripts/ingest.py --path reference_docs/smoke_corpus --chunk-size 1000 --chunk-overlap 120`
  - `uv run python scripts/query.py "smoke corpus keyword" --top-k 5 --rebuild-index`

## Recommended next steps

### Week 3 prep
- Add a citation formatting helper (chunk -> citation block) to reuse in answer generation.
- Define a minimal verifier stub that checks “every claim has a citation.”
- Add a small answer-mode path to Streamlit for verified summaries.
