from quantmind.data import generate_regime_series
from quantmind.rag import ResearchCopilot, TfidfRetriever, build_market_corpus


def test_corpus_static_and_live():
    static = build_market_corpus(close=None)
    assert len(static) >= 5

    close = generate_regime_series(n_days=400, seed=3)["close"]
    live = build_market_corpus(close=close)
    ids = {d.id for d in live}
    # Live documents grounded in the actual data are appended.
    assert "live_regime" in ids
    assert "live_backtest" in ids


def test_retriever_ranks_relevant_doc_first():
    retriever = TfidfRetriever().fit(build_market_corpus(close=None))
    hits = retriever.query("why does volatility rise in a bear market", k=3)
    assert hits, "expected at least one hit"
    top_ids = [d.id for d, _ in hits]
    # The volatility/bear document should surface near the top.
    assert "volatility" in top_ids


def test_copilot_offline_is_grounded(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    copilot = ResearchCopilot(docs=build_market_corpus(close=None))
    result = copilot.ask("What is the Sharpe ratio?")

    assert result["used_llm"] is False
    assert result["sources"]
    assert len(result["answer"]) > 0
    # Offline answers cite their grounding documents.
    assert "[" in result["answer"]
