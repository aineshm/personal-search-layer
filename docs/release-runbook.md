# Release Runbook

This runbook defines the repeatable local process for release-candidate validation.

## Preconditions
- Python 3.12 and dependencies installed via `uv`.
- Clean local environment for reproducible runs.

## 1. Environment setup
```bash
uv python install 3.12
uv venv --python 3.12
uv sync
```

## 2. Migrate schema
Always migrate before ingestion/query/eval commands.

```bash
uv run python scripts/maintenance.py --migrate
```

## 3. Ingest release validation corpus
Use your target local corpus path for release checks.

```bash
uv run python scripts/ingest.py --path reference_docs/smoke_corpus --chunk-size 1000 --chunk-overlap 120
```

## 4. Build/refresh index and smoke query
```bash
uv run python scripts/query.py "smoke corpus keyword" --mode search --top-k 8 --rebuild-index
```

## 5. Run static quality checks
```bash
uv run ruff check .
uv run pytest -q
uv run pytest -q -m slow
```

## 6. Run retrieval eval gate
```bash
uv run python eval/run_golden_eval.py --top-k 5 --rebuild-index
uv run python eval/summarize_eval.py --report-path eval/reports/latest.json
```

Interpretation:
- Check `metrics@k` for lexical/vector/hybrid.
- Verify hybrid retrieval quality is not severely regressed from prior baseline.
- Inspect `metrics_delta` for significant drops.

## 7. Run answer eval gate
```bash
uv run python eval/run_answer_eval.py --report-path eval/reports/answer_latest.json
```

Interpretation:
- Review `metrics` and `gates` fields.
- Key tracked metrics:
  - `citation_coverage`
  - `citation_precision_proxy`
  - `abstain_correctness`
  - `conflict_correctness`
  - `repair_rate`
  - `false_repair_rate`
- `false_answer_rate`
- `over_abstain_rate`
- `under_abstain_rate`
- `unsupported_claim_rate`

Pass/fail matrix (phase gates):
- `abstain_correctness >= 0.95` must pass.
- `citation_coverage >= 0.90` must pass.
- `conflict_correctness >= 0.85` must pass.
- `false_repair_rate <= 0.20` must pass.
- `gates.overall_pass` is true only when all four conditions pass.

## 8. Validate query UX paths
```bash
uv run python scripts/query.py "smoke corpus keyword" --mode search
uv run python scripts/query.py "smoke corpus keyword" --mode answer
uv run python scripts/query.py "what is the orbital period of kepler-186f" --mode answer --skip-vector
```

Expected:
- Search mode returns ranked evidence.
- Answer mode returns cited claims for in-corpus queries.
- Out-of-corpus query abstains with searched-query rationale.

## 9. Baseline locking process
For a release candidate:
1. Keep generated reports under `eval/reports/history/`.
2. Record commit hash + retrieval summary + answer summary in release notes.
3. Treat that snapshot as the comparison baseline for the next cycle.
4. To run against an explicit locked baseline: `--baseline-path <path/to/report.json>`.
5. Baseline updates require a PR note that explains metric deltas and why the new baseline is accepted.

Exception process:
- If a release must proceed with a failed non-primary gate, document:
  - failed gate(s),
  - risk assessment,
  - mitigation plan and due date.
- Primary trust gate (`abstain_correctness`) has no exception by default.

## 10. Failure handling
- Schema errors: rerun `scripts/maintenance.py --migrate`.
- Manifest mismatch/no vector hits: rebuild index with `--rebuild-index`.
- Metric regression: inspect `cases_detail` in both retrieval and answer reports and fix before release.
