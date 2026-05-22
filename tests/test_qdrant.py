import numpy as np

from quantmind.rag import (
    HashingEmbedder,
    QdrantRetriever,
    ResearchCopilot,
    build_market_corpus,
)


def test_hashing_embedder_is_deterministic_and_normalised():
    emb = HashingEmbedder(dim=128)
    a = emb.embed(["bear market volatility"])
    b = emb.embed(["bear market volatility"])
    assert a.shape == (1, 128)
    np.testing.assert_allclose(a, b)  # deterministic
    assert np.isclose(np.linalg.norm(a[0]), 1.0, atol=1e-5)  # L2-normalised


def test_qdrant_retriever_embedded_mode():
    retriever = QdrantRetriever().fit(build_market_corpus(close=None))
    hits = retriever.query("why does volatility rise in a bear market", k=3)
    assert hits
    assert "volatility" in [d.id for d, _ in hits]
    # Cosine scores are bounded.
    assert all(-1.01 <= score <= 1.01 for _, score in hits)


def test_copilot_with_qdrant_backend(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    copilot = ResearchCopilot(docs=build_market_corpus(close=None), retriever=QdrantRetriever())
    result = copilot.ask("What is the Sharpe ratio?")
    assert result["used_llm"] is False
    assert result["sources"]
    assert len(result["answer"]) > 0
