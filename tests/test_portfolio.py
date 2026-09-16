import pytest
import time

import numpy as np
import pandas as pd

from quantmind.portfolio import assess_portfolio, assess_resilience, normalize_holdings
from quantmind.portfolio import health


def test_normalize_holdings_merges_and_normalizes_weights():
    result = normalize_holdings([
        {"symbol": "spy", "weight": 60}, {"symbol": "SPY", "weight": 20}, {"symbol": "tlt", "weight": 20},
    ])
    assert result == [{"symbol": "SPY", "weight": 0.8}, {"symbol": "TLT", "weight": 0.2}]


def test_synthetic_portfolio_assessment_returns_real_diagnostics():
    result = assess_portfolio([
        {"symbol": "A", "weight": 0.8}, {"symbol": "B", "weight": 0.2},
    ], source="synthetic")
    assert result["observations"] > 500
    assert result["summary"]["top_holding_weight"] == 0.8
    assert result["summary"]["effective_holdings"] == pytest.approx(1.47, abs=0.01)
    assert any(flag["code"] == "single_holding_concentration" for flag in result["flags"])


def test_resilience_lab_is_reproducible_and_attributes_all_portfolio_risk():
    result = assess_resilience([
        {"symbol": "A", "weight": 0.8}, {"symbol": "B", "weight": 0.2},
    ], source="synthetic", simulations=256, horizon_days=20)
    current = result["current_allocation"]
    assert result["simulation"]["method"].startswith("5-day moving-block")
    assert len(current["risk_attribution"]) == 2
    assert sum(item["risk_contribution"] for item in current["risk_attribution"]) == pytest.approx(1.0, abs=1e-3)
    assert 0 <= current["probability_of_loss"] <= 1
    assert current["cvar_95_horizon_return"] <= current["p05_horizon_return"]


def test_live_price_loading_is_concurrent_and_preserves_symbol_order(monkeypatch):
    delay = 0.08

    def delayed_loader(symbol, period):
        time.sleep(delay)
        return {"close": pd.Series(100.0 + np.arange(100), name="close")}

    monkeypatch.setattr(health, "load_prices", delayed_loader)
    holdings = [{"symbol": symbol, "weight": 1 / 6} for symbol in ["A", "B", "C", "D", "E", "F"]]
    started = time.perf_counter()
    prices = health._live_prices(holdings, "1y")
    elapsed = time.perf_counter() - started

    # Sequential I/O would take roughly 6 * delay = 0.48 seconds. The bounded
    # pool should finish near one request duration while retaining input order.
    assert elapsed < 0.24
    assert list(prices.columns) == [item["symbol"] for item in holdings]
