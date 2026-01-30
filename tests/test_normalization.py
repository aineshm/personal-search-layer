from personal_search_layer.ingestion.normalization import normalize_text


def test_normalize_text_nfkc_lowercase_and_whitespace() -> None:
    raw = "  Ｆｏｏ\tBAR\nBaz  "
    assert normalize_text(raw) == "foo bar baz"
