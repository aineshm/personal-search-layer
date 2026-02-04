# Implementation Plan

This plan tracks the current build and the next milestones. Each milestone includes high-level bullets and acceptance criteria.

## Week 1: Ingestion + indexes + search-only UI + logging

### Goals
- Local-first ingestion and search pipeline.
- Deterministic indexing and repeatable retrieval.
- Search-only UI for evidence exploration.

### Deliverables
- Ingestion pipeline (loaders, normalization, chunking).
- SQLite schema with FTS5 lexical index.
- FAISS vector index build.
- Hybrid retrieval (RRF).
- Search-only UI.
- Telemetry for ingest/query/UI.
- Smoke corpus and basic tests.

### Acceptance criteria
- `scripts/ingest.py` ingests `reference_docs/smoke_corpus` with non-zero chunks.
- `scripts/query.py` returns at least one hit for "smoke corpus keyword".
- `src/personal_search_layer/ui.py` runs and returns results.
- `pytest -q` passes for existing tests.
- Logs include latency metrics for ingest and query.

## Week 2: Hybrid retrieval + router agent + baseline evals

### Goals
- Make retrieval behavior testable and measurable.
- Add intent routing for better parameterization.

### Deliverables
- Query router agent (intent classification, pipeline params).
- Eval harness for retrieval and routing.
- Golden retrieval set with expected evidence IDs.
- Metrics: Recall@10, MRR, nDCG@10; router accuracy.
- Minimal CLI to run evals and report metrics.

### Acceptance criteria
- Router returns stable intent labels for test cases.
- Retrieval eval suite runs locally in < 2 minutes.
- Baseline metrics are logged and stored with a version hash.
- Hybrid recall improves or matches lexical-only for paraphrase queries.

### Notes
- When using sentence-transformers, first run will download model weights locally.

## Week 3: Agentic RAG MVP (multi-hop + verifier/repair)

### Goals
- Bound multi-hop and verification loops.
- Enforce evidence-only answers with citations and abstain.

### Deliverables
- Multi-hop retrieval (max 1 hop).
- Verifier/repair pass (max 1 repair pass).
- Claim-by-claim citation formatting.
- Conflict detection and abstain rationale.

### Acceptance criteria
- For in-corpus queries, answers include citations for every claim.
- For out-of-corpus queries, system abstains and lists queries searched.
- Multi-hop only triggers once and is logged in tool traces.
- Verifier detects missing citations and either repairs or abstains.

## Ongoing: Engineering guardrails

### Goals
- Deterministic, testable tools.
- Keep workflows local-only.

### Acceptance criteria
- Tool functions callable without LLM.
- No cloud calls in default workflows.
- New behavior changes include tests or eval cases.

## Risk register (current)
- FAISS index and SQLite mapping must stay aligned.
- PDF parsing can be lossy; extraction failures should be logged.
- Deterministic hash embeddings are placeholders; quality is limited.

## Planned extensions
- Replace hash embeddings with local model embeddings.
- Reranker integration for hybrid results.
- Persistent cache for retrieval results.
- UI: Answer mode with citation view and conflict display.
