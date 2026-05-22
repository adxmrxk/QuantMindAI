"""Regime update stream — the payloads pushed over the WebSocket / bus."""

from __future__ import annotations

import asyncio

import pandas as pd

from quantmind.models import analyze


def iter_regime_updates(close: pd.Series, n_states: int = 3) -> list[dict]:
    """Compute the regime series once and return per-day update payloads."""
    result = analyze(close, n_states=n_states)
    return [
        {
            "type": "tick",
            "date": idx.strftime("%Y-%m-%d"),
            "close": round(float(row["close"]), 4),
            "label": str(row["label"]),
            "confidence": round(float(row["confidence"]), 4),
        }
        for idx, row in result.frame.iterrows()
    ]


async def regime_stream(close: pd.Series, limit: int = 120, speed: float = 0.05, n_states: int = 3):
    """Async generator yielding the most recent ``limit`` updates.

    ``speed`` is the inter-tick delay in seconds (0 = as fast as possible, used
    in tests). A ``bus`` can be driven by forwarding these to ``bus.publish``.
    """
    updates = iter_regime_updates(close, n_states=n_states)
    tail = updates[-limit:] if limit else updates
    for update in tail:
        yield update
        if speed:
            await asyncio.sleep(speed)
