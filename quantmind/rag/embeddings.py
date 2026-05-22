"""Text embedders for vector retrieval.

The default :class:`HashingEmbedder` is deterministic, offline, and needs no
model download — the feature-hashing trick projects text into a fixed-dim,
L2-normalised dense vector. The production swap is a sentence-transformer, which
implements the same ``embed`` contract; switch with ``get_embedder``.
"""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer


class HashingEmbedder:
    def __init__(self, dim: int = 256) -> None:
        self.dim = dim
        self._vec = HashingVectorizer(
            n_features=dim, alternate_sign=True, norm="l2", stop_words="english"
        )

    def embed(self, texts) -> np.ndarray:
        return self._vec.transform(list(texts)).toarray().astype("float32")


class SentenceTransformerEmbedder:  # pragma: no cover - optional heavy dependency
    """Optional embedder backed by sentence-transformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        self.dim = self._model.get_sentence_embedding_dimension()

    def embed(self, texts) -> np.ndarray:
        return np.asarray(
            self._model.encode(list(texts), normalize_embeddings=True), dtype="float32"
        )


def get_embedder(name: str = "hashing", dim: int = 256):
    if name == "sentence-transformers":
        try:
            return SentenceTransformerEmbedder()
        except ImportError:
            pass  # fall back to the offline hashing embedder
    return HashingEmbedder(dim=dim)
