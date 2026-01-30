# Copilot Instructions — personal-search-layer

## Project goal
Build a local-first unified personal search layer with:
- hybrid retrieval (lexical + embeddings)
- agentic routing + multi-hop retrieval (bounded)
- verifier/repair + abstain + conflict handling
- evaluation suite + regression gates

## Code style & conventions
- Python 3.12+; prefer type hints.
- Small, testable functions; avoid hidden global state.
- All answers must be grounded in retrieved chunks (when writing RAG logic).
- Always log tool traces and key metrics (latency, k, rerank used, etc.).
- Prefer laptop-first defaults and env overrides via `config.py`.

## Repo hygiene
- Do NOT commit personal documents, corpora, local DBs, FAISS indexes, model weights, or secrets.
- Prefer deterministic outputs for evaluation (seed where possible).

## Testing expectations
- Add/extend tests when implementing features.
- If changing retrieval/agent behavior, update or add eval cases and explain metric impact.

## Week 1 CLI + maintenance
- Ingest: `scripts/ingest.py` supports `--path`, `--chunk-size`, `--chunk-overlap`, `--max-doc-bytes`,
  `--max-pdf-pages`, `--no-normalize`.
- Query: `scripts/query.py` supports `--top-k`, `--rebuild-index`, `--skip-vector`, `--model-name`, `--dim`, `--rrf-k`.
- Maintenance: `scripts/maintenance.py` supports `--vacuum`, `--integrity-check`, `--backup`.

## Smoke corpus
- `reference_docs/smoke_corpus/` contains tiny synthetic docs for fast ingest/query tests.
- Smoke test runs ingest -> index -> query and expects at least one hit.
