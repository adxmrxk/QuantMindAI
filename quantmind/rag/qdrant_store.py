"""Qdrant-backed vector retriever.

Drop-in replacement for :class:`TfidfRetriever` — same ``fit`` / ``query``
contract — backed by a real Qdrant engine. Defaults to embedded (in-memory)
mode so it runs with no server; point it at a Qdrant URL (``QDRANT_URL`` env or
the ``url`` arg) to use a standalone/clustered instance instead.
"""

from __future__ import annotations

import os

from quantmind.rag.embeddings import HashingEmbedder


class QdrantRetriever:
    def __init__(
        self,
        embedder=None,
        collection: str = "quantmind",
        location: str = ":memory:",
        url: str | None = None,
    ) -> None:
        from qdrant_client import QdrantClient

        self.embedder = embedder or HashingEmbedder()
        self.collection = collection
        self.docs: list = []

        url = url or os.getenv("QDRANT_URL")
        self.client = QdrantClient(url=url) if url else QdrantClient(location=location)

    def fit(self, docs) -> "QdrantRetriever":
        from qdrant_client.models import Distance, PointStruct, VectorParams

        self.docs = list(docs)
        vectors = self.embedder.embed([d.text for d in self.docs])
        dim = int(vectors.shape[1])

        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
        points = [
            PointStruct(
                id=i,
                vector=vectors[i].tolist(),
                payload={"idx": i, "doc_id": d.id, "title": d.title},
            )
            for i, d in enumerate(self.docs)
        ]
        self.client.upsert(collection_name=self.collection, points=points)
        return self

    def query(self, text: str, k: int = 3) -> list[tuple]:
        if not self.docs:
            return []
        qv = self.embedder.embed([text])[0].tolist()
        result = self.client.query_points(collection_name=self.collection, query=qv, limit=k)
        return [(self.docs[p.payload["idx"]], float(p.score)) for p in result.points]
