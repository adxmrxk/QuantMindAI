import numpy as np
import pandas as pd

from quantmind.backtest import (
    BuyAndHold,
    backtest_causal_regime_strategy,
    backtest_regime_strategy,
    metrics,
    run_backtest,
)
from quantmind.data import generate_regime_series


def test_metrics_on_known_series():
    # A constant +0.1%/day series: positive return, zero drawdown.
    r = pd.Series([0.001] * 252)
    assert metrics.total_return(r) > 0
    assert metrics.sharpe_ratio(r) > 0
    assert metrics.max_drawdown(r) == 0.0  # never declines

    # A series that only falls has a drawdown equal to its total loss.
    down = pd.Series([-0.01] * 10)
    assert metrics.max_drawdown(down) < 0
    assert np.isclose(metrics.max_drawdown(down), metrics.total_return(down))


def test_buy_and_hold_equity_tracks_price():
    close = generate_regime_series(n_days=400, seed=3)["close"]
    pos = BuyAndHold().positions(close.to_frame("close"))
    bt = run_backtest(close, pos, cost_bps=0.0)
    # Final equity (growth of $1) equals the price ratio over the window.
    assert np.isclose(bt.equity.iloc[-1], close.iloc[-1] / close.iloc[0], rtol=1e-6)


def test_no_lookahead_first_day_flat():
    close = generate_regime_series(n_days=100, seed=1)["close"]
    pos = BuyAndHold().positions(close.to_frame("close"))
    bt = run_backtest(close, pos)
    # Position is lagged, so day 0 earns nothing.
    assert bt.positions.iloc[0] == 0.0
    assert bt.returns.iloc[0] == 0.0


def test_regime_strategy_sits_out_bear_and_cuts_drawdown():
    close = generate_regime_series(n_days=1500, seed=7)["close"]
    bt, regime = backtest_regime_strategy(close, n_states=3, cost_bps=1.0)

    # Exposure must be zero on Bear days (applied with a one-day lag).
    bear_days = (regime.frame["label"] == "Bear").shift(1).fillna(False)
    assert (bt.positions[bear_days] == 0.0).all()

    # Avoiding the high-volatility bear regime should reduce drawdown.
    assert bt.metrics["max_drawdown"] >= bt.benchmark_metrics["max_drawdown"]
    # All reported metrics are finite numbers.
    assert all(np.isfinite(v) for v in bt.metrics.values())


def test_causal_walk_forward_backtest_has_no_future_observations():
    close = generate_regime_series(n_days=420, seed=7)["close"]
    bt, audit = backtest_causal_regime_strategy(
        close, lookback_days=120, rebalance_days=21, cost_bps=1.0)
    assert audit["future_observations_used"] == 0
    assert audit["rebalances"] > 0
    assert bt.positions.iloc[0] == 0.0
    assert all(np.isfinite(v) for v in bt.metrics.values())
