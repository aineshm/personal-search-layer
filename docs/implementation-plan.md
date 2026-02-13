# Implementation Plan

This plan tracks what is shipped and the remaining quality-hardening milestones.

## Week 1: Ingestion + indexes + UI + logging (completed)

### Delivered
- Ingestion pipeline (loaders, normalization, chunking).
- SQLite + FTS5 schema and storage helpers.
- FAISS index build and embedding mapping.
- Hybrid retrieval (RRF) and initial Streamlit UI.
- Telemetry for ingest/query/UI.

## Week 2: Hybrid retrieval + router + baseline evals (completed)

### Delivered
- Deterministic router with intent-aware pipeline settings.
- Golden retrieval eval harness and report artifacts.
- Router accuracy dataset and tests.
- Retrieval metrics: Recall@K, MRR, nDCG + deltas/history.

## Week 3: Agentic RAG trust MVP (implemented)

### Delivered
- Shared orchestration (`search` and `answer` modes).
- Deterministic extractive claim synthesis with citation spans.
- Verifier with abstain/conflict logic and repair pass.
- Bounded multi-hop expansion (max 1 hop).
- Bounded repair loop (max 1 repair).
- Answer eval harness + verifier/adversarial dataset.

### Acceptance checks (implemented)
- In-corpus answers include claim-level citations.
- Out-of-corpus behavior can abstain with searched-query rationale.
- Tool traces include hop/repair counts and verifier outcomes.
- Tests cover orchestration bounds and answer/verifier flows.

## Foundation hardening (implemented)

### Delivered
- Deterministic doc/chunk IDs and deterministic ingest ordering.
- Active index manifest (`index_manifests`) with snapshot hash.
- Vector retrieval consistency checks against manifest.
- Explicit schema migration path (`scripts/maintenance.py --migrate`).
- Strict schema requirement checks in query/index retrieval paths.
- Router policy externalized to `router_policy.json` (env override supported).

## Remaining milestones (to project-complete quality)

### Milestone A: Answer quality tuning
- Improve citation coverage and abstain correctness on verifier eval.
- Reduce false-repair rate by tightening claim support criteria.

### Milestone B: Eval expansion and gating
- Expand verifier/adversarial cases beyond smoke corpus patterns.
- Keep soft trend gates and enforce severe-regression hard fails.

### Milestone C: Release hardening
- Lock baseline artifacts for retrieval and answer metrics.
- Document release runbook for migration, ingest, eval, and gate interpretation.

## Guardrails
- Local-only by default.
- Deterministic + testable tool behavior.
- Bounded loops (max 1 hop, max 1 repair).
- Every behavior change includes tests and/or eval cases.
