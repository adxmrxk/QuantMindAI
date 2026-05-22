"""Trading strategies.

A strategy maps a feature/regime-annotated frame to a **target exposure** series
in [0, 1] (fraction of capital invested in the asset, remainder in cash). The
engine is responsible for applying yesterday's target to today's return, so
strategies may use information available at the close of each day without
introducing look-ahead bias.
"""

from __future__ import annotations

import pandas as pd

# Default risk posture per regime: fully invested in Bull, half in Neutral,
# in cash during Bear (the high-volatility, negative-drift regime).
DEFAULT_EXPOSURE = {"Bull": 1.0, "Neutral": 0.5, "Bear": 0.0}


class Strategy:
    name = "strategy"

    def positions(self, data: pd.DataFrame) -> pd.Series:  # pragma: no cover - interface
        raise NotImplementedError


class BuyAndHold(Strategy):
    """Always fully invested — the benchmark."""

    name = "Buy & Hold"

    def positions(self, data: pd.DataFrame) -> pd.Series:
        return pd.Series(1.0, index=data.index, name="position")


class RegimeStrategy(Strategy):
    """Size exposure by the detected regime label."""

    name = "Regime-aware"

    def __init__(self, exposures: dict[str, float] | None = None) -> None:
        self.exposures = dict(exposures or DEFAULT_EXPOSURE)

    def positions(self, data: pd.DataFrame) -> pd.Series:
        if "label" not in data.columns:
            raise KeyError("RegimeStrategy requires a 'label' column (run analyze() first).")
        pos = data["label"].map(self.exposures).astype("float64")
        return pos.fillna(0.0).rename("position")
