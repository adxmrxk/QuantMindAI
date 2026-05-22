"""Vectorised backtest engine.

Simulates a long/cash strategy from a target-exposure series, applying
transaction costs on turnover and avoiding look-ahead by lagging positions one
day before they earn returns.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from quantmind.backtest import metrics
from quantmind.backtest.strategies import RegimeStrategy
from quantmind.models import analyze


@dataclass
class BacktestResult:
    returns: pd.Series  # daily strategy returns
    equity: pd.Series  # cumulative growth of $1 (strategy)
    positions: pd.Series  # exposure actually applied each day
    benchmark_returns: pd.Series
    benchmark_equity: pd.Series
    metrics: dict
    benchmark_metrics: dict


def run_backtest(
    close: pd.Series,
    positions: pd.Series,
    cost_bps: float = 1.0,
    periods_per_year: int = 252,
) -> BacktestResult:
    """Simulate a strategy.

    Parameters
    ----------
    close: price series.
    positions: target exposure in [0, 1] per day (same index as ``close``).
    cost_bps: round-trip transaction cost in basis points, charged on turnover.
    """
    close = pd.Series(close, dtype="float64")
    asset_ret = close.pct_change().fillna(0.0)

    target = positions.reindex(close.index).ffill().fillna(0.0).clip(0.0, 1.0)
    # Apply yesterday's target to today's return => no look-ahead.
    applied = target.shift(1).fillna(0.0)
    turnover = applied.diff().abs().fillna(applied.abs())
    costs = turnover * (cost_bps / 1e4)

    strat_ret = (applied * asset_ret - costs).rename("returns")
    bench_ret = asset_ret.rename("benchmark_returns")

    return BacktestResult(
        returns=strat_ret,
        equity=(1.0 + strat_ret).cumprod().rename("equity"),
        positions=applied.rename("position"),
        benchmark_returns=bench_ret,
        benchmark_equity=(1.0 + bench_ret).cumprod().rename("benchmark_equity"),
        metrics=metrics.summary(strat_ret, bench_ret, periods_per_year),
        benchmark_metrics=metrics.summary(bench_ret, None, periods_per_year),
    )


def backtest_regime_strategy(
    close: pd.Series,
    n_states: int = 3,
    exposures: dict[str, float] | None = None,
    cost_bps: float = 1.0,
):
    """Convenience pipeline: detect regimes -> size by regime -> backtest.

    Returns ``(BacktestResult, RegimeResult)``.
    """
    regime = analyze(close, n_states=n_states)
    positions = RegimeStrategy(exposures).positions(regime.frame)
    result = run_backtest(regime.frame["close"], positions, cost_bps=cost_bps)
    return result, regime
