# Evaluation Suite

This directory contains evaluation cases and regression gates for the personal search layer.

## Golden retrieval cases
- `golden_retrieval.jsonl` contains query → expected source mappings for the smoke corpus.
- Use it to validate lexical, vector, and hybrid retrieval quality.

## Run the golden evaluation
The script below computes Recall@K, MRR, and nDCG for each mode (lexical/vector/hybrid).

```bash
uv run python eval/run_golden_eval.py --top-k 5 --rebuild-index
```

By default, the eval run writes a report artifact to `eval/reports/latest.json` with
model metadata and the git commit hash for reproducibility.

The report also includes per-intent and per-case metrics to help diagnose routing and
retrieval quality.

Each run also writes a timestamped report to `eval/reports/history/` and computes
metric deltas versus the previous `latest.json` snapshot.

To view a human-readable summary of the latest report:

```bash
uv run python eval/summarize_eval.py --report-path eval/reports/latest.json
```

## Router intent dataset
- `router_intents.jsonl` contains labeled queries for intent classification.
- Eval reports include a router accuracy summary, and tests enforce a minimum accuracy threshold.

## Answer-mode verifier eval
- `verifier_cases.jsonl` covers supported, out-of-corpus, conflict, and prompt-injection-style cases.
- `run_answer_eval.py` computes citation coverage, citation precision proxy, abstain correctness,
  conflict correctness, repair rate, and false-repair rate.

```bash
uv run python eval/run_answer_eval.py --report-path eval/reports/answer_latest.json
```
