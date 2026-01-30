import numpy as np

from personal_search_layer.retrieval import _filter_faiss_hits


def test_filter_faiss_hits_skips_invalid_indices() -> None:
    mapping = ["chunk-a", "chunk-b"]
    indices = np.array([0, 1, -1, 2])
    scores = np.array([0.9, 0.8, 0.7, 0.6])
    hits = _filter_faiss_hits(indices, scores, mapping)
    assert hits == [(0.9, "chunk-a"), (0.8, "chunk-b")]


def test_filter_faiss_hits_empty_mapping() -> None:
    indices = np.array([0])
    scores = np.array([1.0])
    assert _filter_faiss_hits(indices, scores, []) == []
