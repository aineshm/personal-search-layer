import numpy as np

from personal_search_layer.embeddings import embed_query, embed_texts


def test_hash_embeddings_are_deterministic() -> None:
    vec1 = embed_query("hello", backend="hash", dim=16)
    vec2 = embed_query("hello", backend="hash", dim=16)
    assert np.allclose(vec1, vec2)


def test_hash_embeddings_shape() -> None:
    vectors = embed_texts(["a", "b"], backend="hash", dim=8)
    assert vectors.shape == (2, 8)
