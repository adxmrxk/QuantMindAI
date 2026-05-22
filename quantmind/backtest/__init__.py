"""Backtesting: strategies, performance metrics, and the simulation engine."""

from quantmind.backtest.engine import BacktestResult, backtest_regime_strategy, run_backtest
from quantmind.backtest.strategies import BuyAndHold, RegimeStrategy, Strategy

__all__ = [
    "BacktestResult",
    "run_backtest",
    "backtest_regime_strategy",
    "Strategy",
    "BuyAndHold",
    "RegimeStrategy",
]
