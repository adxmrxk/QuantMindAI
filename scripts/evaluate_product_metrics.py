"""Reproduce QuantMind's product-quality and research-integrity metrics."""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from quantmind.backtest import backtest_causal_regime_strategy, backtest_regime_strategy
from quantmind.data import generate_regime_series
from quantmind.forecast import train_forecaster
from quantmind.portfolio import assess_portfolio, assess_resilience
from quantmind.portfolio import health as portfolio_health


def controlled_six_ticker_latency() -> dict:
    """Measure the exact independent-I/O bottleneck that concurrent loading fixes.

    This is intentionally a controlled benchmark, not a claim about a specific
    external market-data provider. Six equal 120 ms loaders model a six-ticker
    portfolio request so the before and after behavior is repeatable.
    """
    delay = 0.12
    holdings = [{"symbol": symbol, "weight": 1 / 6} for symbol in "ABCDEF"]

    def delayed_loader(symbol: str, period: str) -> dict:
        time.sleep(delay)
        return {"close": pd.Series(100.0 + np.arange(100), name="close")}

    started = time.perf_counter()
    for item in holdings:
        delayed_loader(item["symbol"], "1y")
    sequential = time.perf_counter() - started
    original_loader = portfolio_health.load_prices
    portfolio_health.load_prices = delayed_loader
    try:
        started = time.perf_counter()
        prices = portfolio_health._live_prices(holdings, "1y")
        concurrent = time.perf_counter() - started
    finally:
        portfolio_health.load_prices = original_loader
    return {
        "controlled_six_ticker_sequential_ms": round(sequential * 1000, 1),
        "concurrent_current_ms": round(concurrent * 1000, 1),
        "latency_reduction_percent": round((1 - concurrent / sequential) * 100, 1),
        "symbols_preserved": list(prices.columns),
        "note": "Controlled I/O benchmark. Real provider latency varies.",
    }


def main() -> None:
    close = generate_regime_series(n_days=1500, seed=7)["close"]
    research, _ = backtest_regime_strategy(close, cost_bps=1.0)
    causal, audit = backtest_causal_regime_strategy(close, cost_bps=1.0)
    _, forecast = train_forecaster(close, epochs=12, seed=0)
    concentrated = assess_portfolio(
        [{"symbol": "SPY", "weight": 0.8}, {"symbol": "TLT", "weight": 0.2}], source="synthetic"
    )
    resilience = assess_resilience(
        [{"symbol": "SPY", "weight": 0.8}, {"symbol": "TLT", "weight": 0.2}],
        source="synthetic", simulations=2000, horizon_days=20,
    )
    print(json.dumps({
        "walk_forward_integrity": {
            "before_full_history_future_observations": len(close),
            "after_causal_future_observations": audit["future_observations_used"],
            "causal_rebalances": audit["rebalances"],
            "research_sharpe": research.metrics["sharpe"],
            "causal_sharpe": causal.metrics["sharpe"],
        },
        "forecast": forecast,
        "portfolio_checkup": {
            "before_risk_dimensions": 0,
            "after_risk_dimensions": 6,
            "concentrated_portfolio_score": concentrated["summary"]["diversification_score"],
            "concentration_flagged": any(f["code"] == "single_holding_concentration" for f in concentrated["flags"]),
        },
        "resilience_explainability": {
            "before_holding_level_risk_attribution": 0,
            "after_holding_level_risk_attribution": len(resilience["current_allocation"]["risk_attribution"]),
            "attribution_total": round(sum(
                item["risk_contribution"] for item in resilience["current_allocation"]["risk_attribution"]
            ), 4),
            "simulated_paths": resilience["simulation"]["simulations"],
        },
        "live_portfolio_latency": controlled_six_ticker_latency(),
    }, indent=2))


if __name__ == "__main__":
    main()
