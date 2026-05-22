"""Synthetic regime-switching price generator.

Used for two things:

1. Offline demos — the dashboard and API work with zero network access.
2. Testing — we generate a series with a *known* regime sequence so we can
   measure how well the model recovers it.

The generator is a regime-switching geometric random walk: each day belongs to
one of three latent regimes (bear / neutral / bull), each with its own drift
and volatility, and regimes persist for a while before switching.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Regime parameters expressed as *daily* log-return drift and volatility.
# Index order is meaningful: 0 = Bear, 1 = Neutral, 2 = Bull.
# Volatility is distinct per regime (the realistic stylised fact that down
# markets are more turbulent), which is what lets the model separate them.
_REGIMES = {
    0: {"name": "Bear", "drift": -0.0018, "vol": 0.030},
    1: {"name": "Neutral", "drift": 0.0001, "vol": 0.014},
    2: {"name": "Bull", "drift": 0.0013, "vol": 0.0065},
}


@dataclass(frozen=True)
class SyntheticConfig:
    n_days: int = 1500
    start_price: float = 100.0
    # Expected regime duration in trading days (geometric switching).
    mean_duration: int = 60
    seed: int = 7


def generate_regime_series(
    n_days: int = 1500,
    start_price: float = 100.0,
    mean_duration: int = 60,
    seed: int = 7,
) -> pd.DataFrame:
    """Generate a synthetic OHLC-style series with a known regime label.

    Returns a DataFrame indexed by business days with columns:
    ``close`` and ``true_regime`` (the integer regime that produced each day).
    """
    rng = np.random.default_rng(seed)
    switch_prob = 1.0 / max(mean_duration, 1)

    regimes = np.empty(n_days, dtype=int)
    state = 1  # start Neutral
    for i in range(n_days):
        if i > 0 and rng.random() < switch_prob:
            # Switch to a different regime, biased toward adjacent states.
            choices = [r for r in _REGIMES if r != state]
            state = int(rng.choice(choices))
        regimes[i] = state

    drifts = np.array([_REGIMES[r]["drift"] for r in regimes])
    vols = np.array([_REGIMES[r]["vol"] for r in regimes])
    log_returns = drifts + vols * rng.standard_normal(n_days)

    close = start_price * np.exp(np.cumsum(log_returns))
    index = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n_days)

    return pd.DataFrame({"close": close, "true_regime": regimes}, index=index).rename_axis("date")


def regime_name(regime: int) -> str:
    """Human-readable name for a synthetic regime index."""
    return _REGIMES[regime]["name"]
