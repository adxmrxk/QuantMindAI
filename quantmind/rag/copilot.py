"""The research copilot: retrieve grounding documents, then answer.

Generation uses Claude when ``ANTHROPIC_API_KEY`` is present (with prompt
caching on the system instruction), otherwise an extractive answer composed from
the retrieved documents. Either way the answer is grounded and the sources are
returned, so it never free-associates.
"""

from __future__ import annotations

import os

import pandas as pd

from quantmind.rag.corpus import build_market_corpus
from quantmind.rag.retriever import TfidfRetriever

DEFAULT_MODEL = os.getenv("QUANTMIND_LLM_MODEL", "claude-sonnet-4-6")

_SYSTEM = (
    "You are QuantMind's quantitative research copilot. Answer the user's question "
    "using ONLY the provided context documents. Cite the documents you use by their "
    "[id]. If the context is insufficient to answer, say so plainly. Be concise and "
    "precise; do not give investment advice."
)


class ResearchCopilot:
    def __init__(
        self,
        docs=None,
        close: pd.Series | None = None,
        model: str | None = None,
        retriever=None,
    ) -> None:
        self.docs = list(docs) if docs is not None else build_market_corpus(close)
        # Any object with a ``fit(docs)`` / ``query(text, k)`` contract works
        # (Tf-IDF by default, Qdrant vector search when injected).
        self.retriever = (retriever or TfidfRetriever()).fit(self.docs)
        self.model = model or DEFAULT_MODEL

    def ask(self, query: str, k: int = 3) -> dict:
        hits = self.retriever.query(query, k=k)
        sources = [{"id": d.id, "title": d.title, "score": round(score, 3)} for d, score in hits]
        context = "\n\n".join(f"[{d.id}] {d.title}: {d.text}" for d, _ in hits)

        if hits and os.getenv("ANTHROPIC_API_KEY"):
            answer, used_llm = self._llm_answer(query, context), True
        else:
            answer, used_llm = self._extractive_answer(hits), False

        return {"query": query, "answer": answer, "sources": sources, "used_llm": used_llm}

    @staticmethod
    def _extractive_answer(hits) -> str:
        if not hits:
            return (
                "I don't have grounding documents to answer that. Load market data "
                "first, or rephrase the question."
            )
        ids = ", ".join(f"[{d.id}]" for d, _ in hits[:2])
        body = " ".join(d.text for d, _ in hits[:2])
        return f"{body} (Grounded in {ids}.)"

    def _llm_answer(self, query: str, context: str) -> str:
        import anthropic

        client = anthropic.Anthropic()
        message = client.messages.create(
            model=self.model,
            max_tokens=600,
            system=[{"type": "text", "text": _SYSTEM, "cache_control": {"type": "ephemeral"}}],
            messages=[
                {"role": "user", "content": f"Context documents:\n{context}\n\nQuestion: {query}"}
            ],
        )
        return "".join(block.text for block in message.content if getattr(block, "type", "") == "text")
