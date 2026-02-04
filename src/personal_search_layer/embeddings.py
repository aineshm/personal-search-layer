"""Embedding utilities for local vector search."""

from __future__ import annotations

import hashlib
from functools import lru_cache
from typing import Iterable

import numpy as np

from personal_search_layer.config import EMBEDDING_BACKEND, EMBEDDING_DIM, MODEL_NAME


def embed_texts(
    texts: Iterable[str],
    *,
    backend: str = EMBEDDING_BACKEND,
    model_name: str = MODEL_NAME,
    dim: int | None = None,
) -> np.ndarray:
    text_list = [text for text in texts]
    if not text_list:
        return np.zeros((0, dim or EMBEDDING_DIM), dtype="float32")
    if backend == "hash":
        return _hash_embed_texts(text_list, dim or EMBEDDING_DIM)
    if backend == "sentence-transformers":
        model = _load_sentence_transformer(model_name)
        vectors = model.encode(text_list, normalize_embeddings=True)
        return np.asarray(vectors, dtype="float32")
    raise ValueError(f"Unsupported embedding backend: {backend}")


def embed_query(
    text: str,
    *,
    backend: str = EMBEDDING_BACKEND,
    model_name: str = MODEL_NAME,
    dim: int | None = None,
) -> np.ndarray:
    vectors = embed_texts([text], backend=backend, model_name=model_name, dim=dim)
    return vectors[0]


def get_embedding_dim(
    *,
    backend: str = EMBEDDING_BACKEND,
    model_name: str = MODEL_NAME,
    dim: int | None = None,
) -> int:
    if backend == "hash":
        return dim or EMBEDDING_DIM
    if backend == "sentence-transformers":
        model = _load_sentence_transformer(model_name)
        return int(model.get_sentence_embedding_dimension())
    raise ValueError(f"Unsupported embedding backend: {backend}")


def _hash_to_vector(text: str, dim: int) -> np.ndarray:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:4], "little")
    rng = np.random.default_rng(seed)
    vec = rng.normal(size=(dim,)).astype("float32")
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm


def _hash_embed_texts(texts: Iterable[str], dim: int) -> np.ndarray:
    vectors = [_hash_to_vector(text, dim) for text in texts]
    return np.vstack(vectors) if vectors else np.zeros((0, dim), dtype="float32")


@lru_cache(maxsize=2)
def _load_sentence_transformer(model_name: str):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "sentence-transformers is not installed; run 'uv sync --extra dev'"
        ) from exc
    return SentenceTransformer(model_name)
