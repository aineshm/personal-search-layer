from personal_search_layer.retrieval import _to_fts5_query


def test_to_fts5_query_handles_symbols_and_hyphens() -> None:
    parsed = _to_fts5_query("what is kepler-186f orbital period?")
    assert '"kepler"' in parsed
    assert '"186f"' in parsed
    assert "OR" in parsed


def test_to_fts5_query_empty_when_no_tokens() -> None:
    assert _to_fts5_query("--- !!!") == ""
