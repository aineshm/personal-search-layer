"""Streamlit search-only UI."""

from __future__ import annotations

import html
import re
import sys
import time
from pathlib import Path

import streamlit as st

try:
    from personal_search_layer.retrieval import fuse_hybrid, search_lexical, search_vector
    from personal_search_layer.telemetry import configure_logging, log_event
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / "src"))
    from personal_search_layer.retrieval import (  # type: ignore[reportMissingImports]
        fuse_hybrid,
        search_lexical,
        search_vector,
    )
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
    st.caption("Search-only mode: hybrid retrieval with evidence view.")
    st.info("Local-only: no cloud calls. Evidence view only.")

    query = st.text_input("Search query")
    top_k = st.slider("Top K", min_value=3, max_value=15, value=8)
    use_vector = st.checkbox("Use vector retrieval", value=True)

    if st.button("Search") and query:
        start = time.perf_counter()
        lexical = search_lexical(query, k=top_k)
        vector = search_vector(query, k=top_k) if use_vector else None
        hybrid = fuse_hybrid(lexical, vector, k=top_k) if vector else lexical
        total_latency_ms = (time.perf_counter() - start) * 1000
        log_event(
            logger,
            "search",
            query=query,
            lexical_latency_ms=lexical.latency_ms,
            vector_latency_ms=vector.latency_ms if vector else None,
            fused_latency_ms=hybrid.latency_ms,
            total_latency_ms=total_latency_ms,
        )

        st.subheader("Hybrid results")
        st.caption(
            f"Latency (ms): lexical {lexical.latency_ms:.1f} | "
            f"vector {vector.latency_ms:.1f} | "
            f"hybrid {hybrid.latency_ms:.1f} | "
            f"total {total_latency_ms:.1f}"
            if vector
            else f"Latency (ms): lexical {lexical.latency_ms:.1f} | total {total_latency_ms:.1f}"
        )
        for idx, chunk in enumerate(hybrid.chunks, start=1):
            st.markdown(f"**{idx}. {chunk.source_path}**")
            st.caption(f"Score: {chunk.score:.4f} | Page: {chunk.page or 'n/a'}")
            st.markdown(
                _highlight_terms(chunk.chunk_text, query), unsafe_allow_html=True
            )


if __name__ == "__main__":
    run()
