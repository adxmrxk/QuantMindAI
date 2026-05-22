"""TF-IDF retrieval over the document corpus.

A deliberately dependency-light retriever (scikit-learn TF-IDF + cosine
similarity). The production upgrade path is sentence-transformer embeddings in a
Qdrant vector store; the ``query`` contract here is the same, so swapping the
backend doesn't touch the copilot.
"""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


class TfidfRetriever:
    def __init__(self) -> None:
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.docs: list = []
        self._matrix = None

    def fit(self, docs) -> "TfidfRetriever":
        self.docs = list(docs)
        self._matrix = self.vectorizer.fit_transform([d.text for d in self.docs])
        return self

    def query(self, text: str, k: int = 3) -> list[tuple]:
        """Return up to ``k`` (Document, score) pairs ranked by cosine similarity."""
        if self._matrix is None or not self.docs:
            return []
        q = self.vectorizer.transform([text])
        sims = linear_kernel(q, self._matrix).ravel()
        ranked = sims.argsort()[::-1][:k]
        return [(self.docs[i], float(sims[i])) for i in ranked if sims[i] > 0.0]
