from personal_search_layer.multihop import propose_followup_query


def test_propose_followup_query_uses_missing_claims() -> None:
    followup = propose_followup_query(
        "project alpha timeline",
        None,
        ["Project alpha is 2025 according to source B"],
    )
    assert followup is not None
    assert "2025" in followup


def test_propose_followup_query_none_when_no_signal() -> None:
    assert propose_followup_query("simple query", None, []) is None
