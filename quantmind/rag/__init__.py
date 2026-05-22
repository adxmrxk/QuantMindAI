"""Retrieval-augmented research copilot.

Answers natural-language questions about the market by retrieving relevant
documents — a mix of static market knowledge and *live* facts computed from the
platform (current regime, regime statistics, backtest results) — and grounding
the answer in them. Generation uses Claude when ``ANTHROPIC_API_KEY`` is set,
and falls back to an extractive grounded answer offline.
"""

from quantmind.rag.copilot import ResearchCopilot
from quantmind.rag.corpus import Document, build_market_corpus
from quantmind.rag.embeddings import HashingEmbedder, get_embedder
from quantmind.rag.qdrant_store import QdrantRetriever
from quantmind.rag.retriever import TfidfRetriever

__all__ = [
    "ResearchCopilot",
    "Document",
    "build_market_corpus",
    "TfidfRetriever",
    "QdrantRetriever",
    "HashingEmbedder",
    "get_embedder",
]
