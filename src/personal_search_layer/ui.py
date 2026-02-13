"""Streamlit UI for search and deterministic answer modes."""

from __future__ import annotations

import html
import re
import sys
from pathlib import Path

import streamlit as st

try:
    from personal_search_layer.orchestration import run_query
    from personal_search_layer.telemetry import configure_logging, log_event
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / "src"))
    from personal_search_layer.orchestration import run_query  # type: ignore[reportMissingImports]
    from personal_search_layer.telemetry import (  # type: ignore[reportMissingImports]
        configure_logging,
        log_event,
    )


def _highlight_terms(text: str, query: str) -> str:
    terms = [term for term in re.split(r"\s+", query.strip()) if term]
    if not terms:
        return html.escape(text)
    escaped = html.escape(text)
    for term in sorted(set(terms), key=len, reverse=True):
        pattern = re.compile(re.escape(html.escape(term)), re.IGNORECASE)
        escaped = pattern.sub(r"<mark>\g<0></mark>", escaped)
    return escaped


def run() -> None:
    logger = configure_logging()
    st.set_page_config(page_title="Personal Search Layer", layout="wide")
    st.title("Personal Search Layer")
    st.caption("Local-first retrieval with verified answer mode.")

    query = st.text_input("Query")
    mode = st.radio("Mode", ["search", "answer"], horizontal=True)
    top_k = st.slider("Top K", min_value=3, max_value=20, value=8)
    skip_vector = st.checkbox("Lexical only (skip vector retrieval)", value=False)

    if st.button("Run") and query:
        result = run_query(query, mode=mode, top_k=top_k, skip_vector=skip_vector)
        log_event(
            logger,
            "ui_query",
            query=query,
            mode=mode,
            intent=result.intent,
            top_k=top_k,
            skip_vector=skip_vector,
            tool_trace=result.tool_trace,
            total_latency_ms=result.latency_ms,
        )

        st.caption(f"Intent: {result.intent} | Latency: {result.latency_ms:.1f} ms")

        if mode == "search":
            st.subheader("Results")
            for idx, chunk in enumerate(result.chunks, start=1):
                st.markdown(f"**{idx}. {chunk.source_path}**")
                st.caption(f"Score: {chunk.score:.4f} | Page: {chunk.page or 'n/a'}")
                st.markdown(
                    _highlight_terms(chunk.chunk_text, query), unsafe_allow_html=True
                )
            return

        st.subheader("Answer")
        draft = result.draft_answer
        verification = result.verification
        if not draft:
            st.warning("No draft answer generated.")
            return

        if verification and verification.abstain:
            st.error("Abstained")
            st.write(verification.abstain_reason or "Insufficient evidence.")
            if verification.searched_queries:
                st.caption(
                    "Searched queries: " + " | ".join(verification.searched_queries)
                )
            if verification.conflicts:
                st.write("Conflicts:")
                for conflict in verification.conflicts:
                    st.markdown(f"- {conflict}")
            return

        st.markdown(draft.answer_text)
        st.subheader("Claims and citations")
        for claim in draft.claims:
            st.markdown(f"**{claim.claim_id}.** {claim.text}")
            for citation in claim.citations:
                page = citation.page if citation.page is not None else "n/a"
                st.caption(
                    "source={source} | page={page} | span={start}:{end}".format(
                        source=citation.source_path,
                        page=page,
                        start=citation.quote_span_start,
                        end=citation.quote_span_end,
                    )
                )

        if verification and verification.conflicts:
            st.subheader("Conflicts")
            for conflict in verification.conflicts:
                st.markdown(f"- {conflict}")


if __name__ == "__main__":
    run()
