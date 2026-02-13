import numpy as np

import personal_search_layer.embeddings as embeddings


class _DummySentenceTransformer:
    def __init__(self, dim: int = 8) -> None:
        self._dim = dim

    def encode(self, texts: list[str], normalize_embeddings: bool = True) -> np.ndarray:
        vectors = []
        for text in texts:
            seed = sum(ord(char) for char in text) % 2**32
            rng = np.random.default_rng(seed)
            vec = rng.normal(size=(self._dim,)).astype("float32")
            if normalize_embeddings:
                norm = np.linalg.norm(vec)
                if norm != 0:
                    vec = vec / norm
            vectors.append(vec)
        return np.vstack(vectors)

    def get_sentence_embedding_dimension(self) -> int:
        return self._dim


def test_sentence_transformer_embeddings_are_deterministic(monkeypatch) -> None:
    monkeypatch.setattr(
        embeddings,
        "_load_sentence_transformer",
        lambda model_name, revision=None: _DummySentenceTransformer(),
    )
    vec1 = embeddings.embed_query("hello", backend="sentence-transformers")
    vec2 = embeddings.embed_query("hello", backend="sentence-transformers")
    assert np.allclose(vec1, vec2)


def test_sentence_transformer_embeddings_shape(monkeypatch) -> None:
    monkeypatch.setattr(
        embeddings,
        "_load_sentence_transformer",
        lambda model_name, revision=None: _DummySentenceTransformer(dim=6),
    )
    vectors = embeddings.embed_texts(["a", "b"], backend="sentence-transformers")
    assert vectors.shape == (2, 6)
