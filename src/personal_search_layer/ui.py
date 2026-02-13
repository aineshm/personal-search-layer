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
    # Lightweight scoped styles for readable evidence/claim cards.
    st.markdown(
        """
        <style>
        .psl-metric {
            background: #f6f8fa;
            border: 1px solid #d0d7de;
            border-radius: 10px;
            padding: 0.75rem 0.9rem;
        }
        .psl-metric .label {
            font-size: 0.78rem;
            color: #57606a;
            text-transform: uppercase;
            letter-spacing: 0.02em;
        }
        .psl-metric .value {
            font-size: 1.1rem;
            font-weight: 600;
            color: #24292f;
        }
        .psl-card {
            border: 1px solid #d8dee4;
            border-radius: 10px;
            padding: 0.7rem 0.9rem;
            margin-bottom: 0.7rem;
            background: #ffffff;
        }
        .psl-source {
            font-size: 0.82rem;
            color: #57606a;
            margin-top: 0.25rem;
        }
        .stAlert p {
            margin-bottom: 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("Personal Search Layer")
    st.caption("Local-first retrieval with bounded, verifier-backed answer mode.")

    with st.sidebar:
        st.subheader("Controls")
        mode = st.radio("Mode", ["search", "answer"], horizontal=True)
        top_k = st.slider("Top K", min_value=3, max_value=20, value=8)
        skip_vector = st.checkbox("Lexical only (skip vector retrieval)", value=False)
        run_clicked = st.button("Run Query", type="primary", use_container_width=True)

    query = st.text_input("Query", placeholder="Ask about your local corpus...")

    if run_clicked and query:
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

        # Keep high-signal run metadata visible at the top for trust/debugging.
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(
                (
                    '<div class="psl-metric"><div class="label">Intent</div>'
                    f'<div class="value">{html.escape(result.intent)}</div></div>'
                ),
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                (
                    '<div class="psl-metric"><div class="label">Mode</div>'
                    f'<div class="value">{html.escape(result.mode)}</div></div>'
                ),
                unsafe_allow_html=True,
            )
        with c3:
            st.markdown(
                (
                    '<div class="psl-metric"><div class="label">Latency</div>'
                    f'<div class="value">{result.latency_ms:.1f} ms</div></div>'
                ),
                unsafe_allow_html=True,
            )

        if mode == "search":
            st.subheader("Retrieved Evidence")
            for idx, chunk in enumerate(result.chunks, start=1):
                page = chunk.page if chunk.page is not None else "n/a"
                st.markdown(
                    (
                        '<div class="psl-card">'
                        f"<strong>{idx}. {html.escape(chunk.source_path)}</strong>"
                        f'<div class="psl-source">score={chunk.score:.4f} | page={page}</div>'
                        f"<div>{_highlight_terms(chunk.chunk_text, query)}</div>"
                        "</div>"
                    ),
                    unsafe_allow_html=True,
                )
            return

        st.subheader("Verified Answer")
        draft = result.draft_answer
        verification = result.verification
        if not draft:
            st.warning("No draft answer generated.")
            return

        if verification and verification.abstain:
            st.error("Abstained: insufficient trustworthy evidence")
            st.info(verification.abstain_reason or "Insufficient evidence.")
            if verification.decision_path:
                st.caption("Decision path: " + " -> ".join(verification.decision_path))
            if verification.searched_queries:
                st.caption(
                    "Searched queries: " + " | ".join(verification.searched_queries)
                )
            if verification.conflicts:
                st.subheader("Conflicts")
                for conflict in verification.conflicts:
                    st.markdown(f"- {conflict}")
            return

        # Separate reading surface from evidence/trace to reduce visual noise.
        tabs = st.tabs(["Answer", "Claims & Citations", "Trace"])
        with tabs[0]:
            st.markdown(draft.answer_text)
        with tabs[1]:
            for claim in draft.claims:
                with st.expander(f"{claim.claim_id}: {claim.text}", expanded=True):
                    st.caption(
                        "overlap={:.2f} | supportability={:.2f} | citation_quality={:.2f} | sources={}".format(
                            claim.overlap_score,
                            claim.supportability_score,
                            claim.citation_span_quality,
                            claim.source_count,
                        )
                    )
                    for citation in claim.citations:
                        page = citation.page if citation.page is not None else "n/a"
                        st.code(
                            "source={source} | page={page} | span={start}:{end}".format(
                                source=citation.source_path,
                                page=page,
                                start=citation.quote_span_start,
                                end=citation.quote_span_end,
                            )
                        )
        with tabs[2]:
            if verification:
                st.caption(
                    "verdict={verdict} | confidence={confidence:.2f}".format(
                        verdict=verification.verdict_code,
                        confidence=verification.confidence,
                    )
                )
                if verification.decision_path:
                    st.caption(
                        "decision path: " + " -> ".join(verification.decision_path)
                    )
            st.json(result.tool_trace)

        if verification and verification.conflicts:
            st.subheader("Conflicts")
            for conflict in verification.conflicts:
                st.markdown(f"- {conflict}")


if __name__ == "__main__":
    run()
