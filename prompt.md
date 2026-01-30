# GitHub Copilot Project Prompt - Unified Personal Search Layer

You are assisting with **Unified Personal Search Layer (Local-First RAG)**.

## Project summary
Local-first personal search + agentic RAG that unifies a user's docs/notes into one query layer. The system prioritizes **hybrid retrieval**, **traceable evidence**, **agentic control with bounded loops**, and **evaluation-first quality**. It should run **entirely on-device by default** (privacy-preserving) with an optional hosted inference path later.

## Core problems addressed
- Fragmented personal knowledge across PDFs, notes, saved links, exports, and project docs.
- Low trust/traceability in answers; users need evidence and conflict visibility.
- Poor quality feedback loops; systems don't measure improvements over time.
- Privacy concerns with cloud-only approaches.

## Differentiators (non-negotiable)
- **Local-first privacy**: keep data on device; no cloud dependence.
- **Evaluation-first**: golden datasets + regression gates are first-class deliverables.
- **Hybrid retrieval by default**: lexical + vector + fusion (with optional reranker).
- **Agentic orchestration**: router -> multi-hop -> verifier/repair, **bounded and logged**.
- **Trust UX**: citations, "why this answer?", conflict flags, abstain rationale.

## Functional modes
- **Search mode**: fast retrieval with source drill-down, no generation required.
- **Answer mode**: RAG answers constrained to retrieved evidence with **claim-by-claim citations**.

## Retrieval choices (must reflect in code/design)
- Lexical and semantic retrieval fail differently; **hybrid is required**.
- **Lexical (FTS5/BM25-ish)** handles exact identifiers and short precise queries.
- **Vector** handles paraphrases and conceptual queries.
- **Fusion**: Reciprocal Rank Fusion (RRF) as a default baseline; optional cross-encoder reranking.
- **Laptop-first**: start with **SQLite + FTS5 + FAISS**, avoid heavy services.

## Proposed architecture (laptop-first)
- **Ingestion**: PDF/text/HTML loaders; normalization; metadata extraction; chunking; hashing for idempotency.
- **Storage**: SQLite for docs/chunks/logs; SQLite FTS5 for lexical index.
- **Vector index**: FAISS for embeddings; embedding metadata in SQLite.
- **Retriever**: hybrid fusion + filters; optional reranker; caching.
- **Agent engine**: router -> retrieve -> (optional multi-hop) -> synthesize -> verify/repair -> present/export.
- **LLM**: open-source model local by default (quantized); evidence-only prompting; citations required.
- **UI**: Streamlit MVP with search-only + answer mode; "why this answer?"; feedback buttons.
- **Eval/CI**: golden eval set; regression tests; adversarial tests; performance profiling.

## Minimal data model
- **documents**: doc_id, source_path, source_type, title, created_at, tags, content_hash
- **chunks**: chunk_id, doc_id, chunk_text, start_offset, end_offset, section, page
- **embeddings**: chunk_id, vector_id, model_name, dim (vector in FAISS; metadata in SQLite)
- **runs/logs**: run_id, query, intent, tool_trace, latency_ms, feedback

## Agentic patterns (bounded + logged)
- **Query Router Agent**: classifies intent (lookup/fact/synthesis/compare/timeline/task) and selects pipeline parameters.
- **Multi-hop Retrieval Agent**: iterative retrieval with sub-queries to gather missing evidence (max 1 hop).
- **Verification/Critic Agent**: claim-by-claim grounding, conflict detection, abstain vs repair; can trigger targeted re-retrieval (max 1 repair).
- **Document Action Agent (optional)**: safe non-destructive actions (export summary/checklist, create note with citations).

### Tool interface contracts (deterministic)
- `search_lexical(query, filters, k) -> SearchResult`
- `search_vector(query, filters, k) -> SearchResult`
- `fuse_hybrid(lexical, vector, k) -> SearchResult`
- `rerank(query, chunks, top_k) -> [ScoredChunk]`
- `synthesize_answer(question, chunks, intent) -> DraftAnswer (claims + citations)`
- `summarize(chunks, max_tokens) -> str`
- `verify(question, draft, chunks) -> VerificationResult`
- `export_markdown(title, content_md) -> filepath`
- `export_pdf(title, content_md) -> filepath`
- `create_note(title, content_md, tags) -> note_id`

## Evaluation and safety (must-have)
- **Golden datasets**
  - Retrieval set (50-150 queries) labeled with expected evidence chunk IDs.
  - Router set (~100 queries) with intent labels.
  - Verifier set (30-60 items) with mis-citations/conflicts/out-of-corpus queries.
- **Metrics**
  - Retrieval: Recall@10, MRR, nDCG@10 (breakdown by intent + source type).
  - Answers: citation coverage, citation precision, abstain correctness, conflict correctness.
  - Agent: router accuracy, multi-hop gain (delta recall), repair rate, false-repair rate.
  - System: p50/p95 latency, ingestion errors, index build time, version hashes.
- **Adversarial tests**
  - Prompt injection in docs must be ignored.
  - Out-of-corpus questions must abstain and show what was searched.
  - Conflicts must cite both and avoid false certainty.

## Milestones (from proposal)
- **End of Week 1**: ingestion + indexes + search-only UI + logging.
- **End of Week 2**: hybrid retrieval + router agent + baseline evals.
- **End of Week 3**: agentic RAG MVP (multi-hop + verifier/repair) + end-to-end eval report.

## Repo setup requirements (pre-dev work)
- **Switch to Python 3.12** (current 3.13 GIL version is not needed).
- Use **uv** for environment and dependency management.
- Keep everything **local-only**; no hosted inference or external data processing.

## Implementation guidance
- Favor **simple, laptop-friendly components**; avoid heavy infra.
- Every new feature should include logging hooks and be evaluable.
- Deterministic tool behaviors; LLM only selects tools + parameters.
- Default to **evidence-only** prompts; block unsupported claims.
- Make UX explainable: "why this answer?", citations, conflicts, abstain rationale.

## What to avoid
- Vector-only or BM25-only retrieval.
- Unbounded agent loops.
- Cloud-only workflows or data uploads by default.
- "Chatbot first" features without evaluation and traceability.
