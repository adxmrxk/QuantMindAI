"""Indicators that feed the regime model.

A regime is defined by its *trend* and its *risk*, so the raw daily return —
which is almost pure noise on any single day — is the wrong signal to classify
on. Instead the model consumes two smoothed features:

- ``mom``: rolling mean of log returns (trend / momentum), in percent.
- ``vol``: rolling standard deviation of log returns (realised volatility),
  in percent.

Smoothing matters twice over: it gives the Gaussian HMM separable emission
distributions, and because consecutive days share most of their window the
features are autocorrelated, which naturally yields sticky (persistent) states.

We also carry the raw daily return ``ret`` through for reporting (per-regime
average return), but it is *not* used as a model feature.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Columns the model trains on. Order matters: column 0 (``mom``) is what the
# regime labeller sorts states by, so Bull/Bear map onto the right states.
FEATURE_COLUMNS = ["mom", "vol"]


def compute_features(close: pd.Series | pd.DataFrame, vol_window: int = 20) -> pd.DataFrame:
    """Build the feature frame from a close-price series.

    Returns a DataFrame indexed like ``close`` with columns ``ret``, ``mom`` and
    ``vol`` and the leading NaN window dropped, ready to hand to the model.
    """
    if isinstance(close, pd.DataFrame):
        close = close["close"]
    close = pd.Series(close, dtype="float64")

    log_ret = np.log(close / close.shift(1)) * 100.0
    mom = log_ret.rolling(vol_window).mean()
    vol = log_ret.rolling(vol_window).std()

    feats = pd.DataFrame({"ret": log_ret, "mom": mom, "vol": vol}, index=close.index)
    return feats.dropna()
