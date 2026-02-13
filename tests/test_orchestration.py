from personal_search_layer.models import ScoredChunk, SearchResult
from personal_search_layer.orchestration import run_query


def test_run_query_bounds_with_answer_mode(monkeypatch) -> None:
    chunks = [
        ScoredChunk(
            chunk_id="chunk-1",
            doc_id="doc-1",
            score=1.0,
            chunk_text="Hybrid retrieval combines lexical and vector signals.",
            source_path="notes.md",
            page=1,
        )
    ]

    def fake_lexical(query: str, k: int = 8) -> SearchResult:
        return SearchResult(query=query, mode="lexical", chunks=chunks, latency_ms=1.0)

    monkeypatch.setattr(
        "personal_search_layer.orchestration.search_lexical", fake_lexical
    )
    monkeypatch.setattr(
        "personal_search_layer.orchestration.search_vector",
        lambda query, k=8: SearchResult(
            query=query, mode="vector", chunks=chunks, latency_ms=1.0
        ),
    )
    monkeypatch.setattr(
        "personal_search_layer.orchestration.fuse_hybrid",
        lambda lexical, vector, k=8, lexical_weight=0.5: SearchResult(
            query=lexical.query, mode="hybrid", chunks=chunks, latency_ms=1.0
        ),
    )

    result = run_query("summarize hybrid retrieval", mode="answer")

    orch = result.tool_trace["orchestration"]
    assert orch["hop_count"] <= 1
    assert orch["repair_count"] <= 1
    assert orch["searched_queries"]
    assert result.draft_answer is not None
    assert result.verification is not None
