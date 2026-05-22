"""Turn a price series into supervised sequence windows for the forecaster.

Target = the regime (HMM state) on the *next* day, so the model learns to
forecast rather than to label the current bar. Features are z-scored using
statistics from the training split only, to avoid leaking test information.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantmind.features import compute_features
from quantmind.models import analyze

# The Transformer gets a richer input than the HMM: the raw return as well as
# the smoothed trend and volatility.
FORECAST_FEATURES = ["ret", "mom", "vol"]


def build_dataset(
    close: pd.Series,
    n_states: int = 3,
    seq_len: int = 30,
    vol_window: int = 20,
):
    """Return (X, y, regime_states).

    X: float32 array (n_windows, seq_len, n_features)
    y: int array (n_windows,) — regime state on the day after each window
    regime_states: the full per-day state series (for reference/labelling)
    """
    feats = compute_features(close, vol_window=vol_window)
    regime = analyze(close, n_states=n_states, vol_window=vol_window)
    states = regime.frame["state"].to_numpy()

    X_feat = feats[FORECAST_FEATURES].to_numpy(dtype="float32")
    windows, targets = [], []
    # Window ends at day t (inclusive); predict the state at day t+1.
    for t in range(seq_len - 1, len(X_feat) - 1):
        windows.append(X_feat[t - seq_len + 1 : t + 1])
        targets.append(states[t + 1])
    X = np.asarray(windows, dtype="float32")
    y = np.asarray(targets, dtype="int64")
    return X, y, states


def zscore_fit(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flat = X.reshape(-1, X.shape[-1])
    mean = flat.mean(axis=0)
    std = flat.std(axis=0) + 1e-8
    return mean.astype("float32"), std.astype("float32")


def zscore_apply(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((X - mean) / std).astype("float32")
