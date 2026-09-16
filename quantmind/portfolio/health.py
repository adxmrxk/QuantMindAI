"""Evidence-based portfolio risk diagnostics, not investment advice.

Self-directed investors often see individual ticker charts but miss the risks
created by a portfolio as a system: concentration, correlated holdings, and
historical drawdown. This module turns a set of holdings into transparent,
reproducible diagnostics and flags. It deliberately reports observations and
thresholds rather than issuing buy/sell instructions.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

from quantmind.backtest import metrics
from quantmind.data import load_prices
from quantmind.graph import generate_multi_asset_returns


def normalize_holdings(holdings: list[dict]) -> list[dict]:
    if not 2 <= len(holdings) <= 10:
        raise ValueError("provide between 2 and 10 holdings")
    merged: dict[str, float] = {}
    for item in holdings:
        symbol = str(item.get("symbol", "")).strip().upper()
        weight = float(item.get("weight", 0.0))
        if not symbol or not symbol.replace("-", "").replace(".", "").isalnum():
            raise ValueError("holding symbols must be non-empty ticker-like identifiers")
        if not np.isfinite(weight) or weight <= 0:
            raise ValueError("holding weights must be positive finite numbers")
        merged[symbol] = merged.get(symbol, 0.0) + weight
    total = sum(merged.values())
    return [{"symbol": symbol, "weight": round(weight / total, 6)} for symbol, weight in merged.items()]


def _synthetic_prices(holdings: list[dict], n_days: int = 756) -> pd.DataFrame:
    returns, _ = generate_multi_asset_returns(n_assets=len(holdings), n_days=n_days, seed=42)
    returns.columns = [item["symbol"] for item in holdings]
    return (1.0 + returns).cumprod() * 100.0


def _live_prices(holdings: list[dict], period: str) -> pd.DataFrame:
    """Load independent ticker histories concurrently, preserving requested order.

    Portfolio calculations need overlapping history from every holding. The old
    implementation waited for each remote market-data request before beginning
    the next one, multiplying user-perceived latency for 6 to 10 holdings.
    A bounded pool makes the independent I/O concurrent without overwhelming a
    data provider or creating an unbounded number of threads.
    """
    symbols = [item["symbol"] for item in holdings]

    def fetch(symbol: str) -> pd.Series:
        return load_prices(symbol, period=period)["close"].rename(symbol)

    fetched: dict[str, pd.Series] = {}
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=min(6, len(symbols)), thread_name_prefix="quantmind-price") as pool:
        pending = {pool.submit(fetch, symbol): symbol for symbol in symbols}
        for future in as_completed(pending):
            symbol = pending[future]
            try:
                fetched[symbol] = future.result()
            except (ValueError, RuntimeError, OSError) as exc:
                failures.append(f"{symbol}: {exc}")
    if failures:
        raise RuntimeError("could not load portfolio history: " + "; ".join(failures))
    series = [fetched[symbol] for symbol in symbols]
    prices = pd.concat(series, axis=1).dropna()
    if len(prices) < 80:
        raise ValueError("holdings do not have enough overlapping price history")
    return prices


def assess_portfolio(
    holdings: list[dict], period: str = "3y", source: str = "auto"
) -> dict:
    """Return transparent historical diagnostics for a weighted portfolio."""
    normalized = normalize_holdings(holdings)
    use_synthetic = source == "synthetic"
    prices = _synthetic_prices(normalized) if use_synthetic else _live_prices(normalized, period)
    returns = prices.pct_change().dropna()
    weights = pd.Series({item["symbol"]: item["weight"] for item in normalized})
    portfolio_returns = returns.mul(weights, axis=1).sum(axis=1)
    correlation = returns.corr()
    off_diagonal = correlation.to_numpy()[~np.eye(len(correlation), dtype=bool)]
    avg_correlation = float(np.mean(off_diagonal)) if off_diagonal.size else 0.0
    hhi = float(np.square(weights).sum())
    effective_holdings = float(1.0 / hhi)
    top = weights.sort_values(ascending=False)
    var_95 = float(-np.quantile(portfolio_returns, 0.05))
    risk_metrics = metrics.summary(portfolio_returns)
    flags = []
    if float(top.iloc[0]) >= 0.50:
        flags.append({"level": "high", "code": "single_holding_concentration",
                      "message": f"{top.index[0]} is {top.iloc[0]:.0%} of the portfolio."})
    if avg_correlation >= 0.70:
        flags.append({"level": "high", "code": "correlation_concentration",
                      "message": f"Average pairwise correlation is {avg_correlation:.2f}."})
    if risk_metrics["max_drawdown"] <= -0.25:
        flags.append({"level": "moderate", "code": "historical_drawdown",
                      "message": f"Historical maximum drawdown was {risk_metrics['max_drawdown']:.1%}."})
    if not flags:
        flags.append({"level": "info", "code": "no_threshold_breach",
                      "message": "No configured concentration or drawdown threshold was breached."})
    # Score represents coverage of transparent risk checks, not an investment rating.
    diversification_score = round(float(np.clip(
        100 * (0.50 * min(effective_holdings / len(weights), 1.0) +
               0.50 * (1.0 - max(avg_correlation, 0.0))), 0, 100)), 1)
    return {
        "source": "synthetic" if use_synthetic else "auto",
        "period": period,
        "holdings": normalized,
        "observations": int(len(returns)),
        "summary": {
            "annualized_volatility": risk_metrics["ann_volatility"],
            "max_drawdown": risk_metrics["max_drawdown"],
            "historical_var_95_daily": round(var_95, 4),
            "average_pairwise_correlation": round(avg_correlation, 3),
            "top_holding_weight": round(float(top.iloc[0]), 4),
            "effective_holdings": round(effective_holdings, 2),
            "diversification_score": diversification_score,
        },
        "flags": flags,
        "correlation": {
            "symbols": list(correlation.columns),
            "matrix": correlation.round(3).values.tolist(),
        },
        "disclaimer": "Historical diagnostic only. It is not investment advice or a forecast of loss.",
    }


def _resilience_summary(
    returns: pd.DataFrame, weights: pd.Series, simulations: int, horizon_days: int
) -> dict:
    """Measure historical risk and a reproducible moving-block bootstrap distribution."""
    values = returns.loc[:, weights.index].to_numpy(dtype=float)
    weight_values = weights.to_numpy(dtype=float)
    daily = values @ weight_values
    covariance = np.cov(values, rowvar=False, ddof=1)
    portfolio_variance = float(weight_values @ covariance @ weight_values)
    if portfolio_variance <= 0:
        raise ValueError("portfolio return history has no usable variance")
    marginal = covariance @ weight_values
    contributions = weight_values * marginal / portfolio_variance

    # Moving blocks preserve each day's cross-asset relationships and short-run
    # return clustering better than independently resampling every asset.
    block_size = min(5, len(values))
    blocks_needed = int(np.ceil(horizon_days / block_size))
    rng = np.random.default_rng(17)
    starts = rng.integers(0, len(values), size=(simulations, blocks_needed))
    offsets = np.arange(block_size)
    indexes = (starts[:, :, None] + offsets[None, None, :]) % len(values)
    sampled = values[indexes].reshape(simulations, -1, len(weights))[:, :horizon_days, :]
    path_returns = np.einsum("shn,n->sh", sampled, weight_values)
    terminal_returns = np.prod(1.0 + path_returns, axis=1) - 1.0
    p05 = float(np.quantile(terminal_returns, 0.05))
    cvar = float(terminal_returns[terminal_returns <= p05].mean())
    annualized_volatility = float(np.std(daily, ddof=1) * np.sqrt(252))
    return {
        "annualized_volatility": round(annualized_volatility, 4),
        "median_horizon_return": round(float(np.median(terminal_returns)), 4),
        "p05_horizon_return": round(p05, 4),
        "cvar_95_horizon_return": round(cvar, 4),
        "probability_of_loss": round(float(np.mean(terminal_returns < 0.0)), 4),
        "probability_of_10pct_loss": round(float(np.mean(terminal_returns <= -0.10)), 4),
        "risk_attribution": [
            {
                "symbol": symbol,
                "weight": round(float(weights[symbol]), 4),
                "risk_contribution": round(float(contributions[index]), 4),
            }
            for index, symbol in enumerate(weights.index)
        ],
    }


def assess_resilience(
    holdings: list[dict], period: str = "3y", source: str = "auto",
    simulations: int = 2000, horizon_days: int = 20,
) -> dict:
    """Explain risk drivers and compare the portfolio with equal weighting.

    This is not an optimizer or a recommendation. It creates a concrete
    counterfactual so a user can see the concentration trade-off in their own
    holdings, using reproducible moving-block bootstrap samples.
    """
    if not 256 <= simulations <= 10000:
        raise ValueError("simulations must be between 256 and 10000")
    if not 5 <= horizon_days <= 60:
        raise ValueError("horizon_days must be between 5 and 60")
    normalized = normalize_holdings(holdings)
    use_synthetic = source == "synthetic"
    prices = _synthetic_prices(normalized) if use_synthetic else _live_prices(normalized, period)
    returns = prices.pct_change().dropna()
    weights = pd.Series({item["symbol"]: item["weight"] for item in normalized})
    equal_weights = pd.Series(1.0 / len(weights), index=weights.index)
    current = _resilience_summary(returns, weights, simulations, horizon_days)
    equal = _resilience_summary(returns, equal_weights, simulations, horizon_days)
    current_concentration = float(np.square(weights).sum())
    equal_concentration = float(np.square(equal_weights).sum())
    return {
        "source": "synthetic" if use_synthetic else "auto",
        "period": period,
        "holdings": normalized,
        "observations": int(len(returns)),
        "simulation": {
            "method": "5-day moving-block bootstrap of joint historical daily returns",
            "seed": 17,
            "simulations": simulations,
            "horizon_days": horizon_days,
        },
        "current_allocation": current,
        "equal_weight_counterfactual": equal,
        "comparison": {
            "annualized_volatility_change": round(
                equal["annualized_volatility"] - current["annualized_volatility"], 4
            ),
            "p05_horizon_return_change": round(
                equal["p05_horizon_return"] - current["p05_horizon_return"], 4
            ),
            "concentration_hhi_change": round(equal_concentration - current_concentration, 4),
        },
        "disclaimer": (
            "Historical simulation and allocation comparison only. It is not investment advice, "
            "a forecast, or a recommendation to rebalance."
        ),
    }
