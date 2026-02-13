# Project Status Summary

_Last updated: 2026-02-13_

## Current state
- Week 1 and Week 2 foundations are complete and passing.
- Week 3 trust-oriented MVP is implemented (search + answer modes with bounded orchestration).
- Foundation hardening refactor is applied (deterministic IDs/order, index manifests, explicit migration path, externalized router policy).

## Implemented capabilities

### Core product behavior
- Search mode: hybrid retrieval (lexical + vector + RRF fusion) with optional rerank.
- Answer mode: deterministic extractive claims with claim-level citations.
- Verification: unsupported-claim detection, conflict detection, abstain rationale, and single repair attempt.
- Bounded loops: max 1 multi-hop expansion and max 1 repair pass.

### Storage/indexing foundations
- Explicit migration support via `scripts/maintenance.py --migrate`.
- Strict schema checks for read/query paths (`require_schema`).
- Deterministic document and chunk identifiers.
- Deterministic ingestion file ordering and chunk retrieval ordering.
- Active index manifest with model/dim/count/snapshot hash metadata.
- Vector retrieval checks manifest consistency before serving FAISS hits.

### Routing and policy
- Router behavior externalized to `src/personal_search_layer/router_policy.json`.
- Optional policy override via `PSL_ROUTER_POLICY`.

### Evaluation suite
- Retrieval eval (`eval/run_golden_eval.py`) with report history and metric deltas.
- Answer eval (`eval/run_answer_eval.py`) with citation/abstain/conflict/repair metrics.

## Validation status
- `uv run ruff check .` passes.
- `uv run pytest -q` passes (`43 passed, 1 deselected`).
- `uv run pytest -q -m slow` passes (`1 passed`).

## Important commands
- Migrate schema: `uv run python scripts/maintenance.py --migrate`
- Ingest corpus: `uv run python scripts/ingest.py --path reference_docs/smoke_corpus`
- Search query: `uv run python scripts/query.py "smoke corpus keyword" --mode search`
- Answer query: `uv run python scripts/query.py "smoke corpus keyword" --mode answer`
- Retrieval eval: `uv run python eval/run_golden_eval.py --top-k 5 --rebuild-index`
- Answer eval: `uv run python eval/run_answer_eval.py --report-path eval/reports/answer_latest.json`

## Remaining quality work
- Improve answer-quality metrics (citation coverage, abstain correctness, repair quality) to consistently meet thresholds.
- Expand verifier/adversarial eval cases for broader corpus patterns.
- Finalize release hardening docs and baseline-lock process for metric gates.
