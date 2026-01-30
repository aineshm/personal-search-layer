# AGENTS.md — personal-search-layer

## Mission
Ship a local-first personal search layer with measurable quality:
hybrid retrieval, agent router, bounded multi-hop, verifier/repair, and eval regression gates.

## Non-negotiables
- Never fabricate citations: every claim must be supported by cited chunk text.
- If evidence is weak or missing, abstain and say what was searched.
- Keep loops bounded: max 1 multi-hop expansion + max 1 repair pass.

## Repo rules
- Never add personal data to git (corpus/, *.db, FAISS indexes, model weights, secrets).
- Prefer minimal dependencies; explain why new deps are needed.
- Use Python 3.12 and manage environments with uv.
- Keep workflows local-only; avoid hosted inference defaults.

## Engineering preferences
- Deterministic + testable tools. Tool functions should be callable without the LLM.
- Separate: ingestion / retrieval / agent-orchestration / eval.
- Add tests or eval cases for any behavior change.

## Commands (update as the repo evolves)
- Run tests: `pytest -q`
- Lint/format: `ruff check .` and `ruff format .`

## Output expectations
- Make small, reviewable commits.
- Include brief notes on metric impact when changing retrieval/agent logic.
