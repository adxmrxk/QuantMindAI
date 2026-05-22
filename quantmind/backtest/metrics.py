"""Performance and risk metrics computed from a daily-returns series.

All functions take a pandas Series of *simple* daily returns (e.g. 0.01 == +1%)
and are robust to NaNs and degenerate (zero-variance) inputs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def _clean(returns) -> pd.Series:
    return pd.Series(returns, dtype="float64").dropna()


def total_return(returns) -> float:
    r = _clean(returns)
    return float((1.0 + r).prod() - 1.0)


def cagr(returns, periods_per_year: int = TRADING_DAYS) -> float:
    r = _clean(returns)
    if len(r) == 0:
        return 0.0
    growth = (1.0 + r).prod()
    years = len(r) / periods_per_year
    if years <= 0 or growth <= 0:
        return 0.0
    return float(growth ** (1.0 / years) - 1.0)


def ann_volatility(returns, periods_per_year: int = TRADING_DAYS) -> float:
    r = _clean(returns)
    return float(r.std(ddof=0) * np.sqrt(periods_per_year))


def sharpe_ratio(returns, rf: float = 0.0, periods_per_year: int = TRADING_DAYS) -> float:
    r = _clean(returns)
    sd = r.std(ddof=0)
    if sd == 0 or np.isnan(sd):
        return 0.0
    excess = r - rf / periods_per_year
    return float(excess.mean() / sd * np.sqrt(periods_per_year))


def sortino_ratio(returns, rf: float = 0.0, periods_per_year: int = TRADING_DAYS) -> float:
    r = _clean(returns)
    downside = r[r < 0]
    dd = downside.std(ddof=0)
    if dd == 0 or np.isnan(dd):
        return 0.0
    excess = r - rf / periods_per_year
    return float(excess.mean() / dd * np.sqrt(periods_per_year))


def max_drawdown(returns) -> float:
    r = _clean(returns)
    if len(r) == 0:
        return 0.0
    # Anchor the running peak at the initial $1 of capital, so a series that
    # only ever loses has a drawdown equal to its total loss.
    equity = pd.concat([pd.Series([1.0]), (1.0 + r).cumprod()], ignore_index=True)
    drawdown = equity / equity.cummax() - 1.0
    return float(drawdown.min())


def beta(returns, benchmark) -> float:
    r, b = _clean(returns), _clean(benchmark)
    idx = r.index.intersection(b.index)
    r, b = r.loc[idx], b.loc[idx]
    var = b.var(ddof=0)
    if var == 0 or np.isnan(var) or len(r) < 2:
        return 0.0
    cov = np.cov(r.to_numpy(), b.to_numpy(), ddof=0)[0, 1]
    return float(cov / var)


def alpha(returns, benchmark, periods_per_year: int = TRADING_DAYS) -> float:
    r, b = _clean(returns), _clean(benchmark)
    idx = r.index.intersection(b.index)
    r, b = r.loc[idx], b.loc[idx]
    if len(r) == 0:
        return 0.0
    return float((r.mean() - beta(r, b) * b.mean()) * periods_per_year)


def summary(returns, benchmark=None, periods_per_year: int = TRADING_DAYS) -> dict:
    """Bundle the common metrics into one rounded dict for APIs/dashboards."""
    out = {
        "total_return": round(total_return(returns), 4),
        "cagr": round(cagr(returns, periods_per_year), 4),
        "ann_volatility": round(ann_volatility(returns, periods_per_year), 4),
        "sharpe": round(sharpe_ratio(returns, 0.0, periods_per_year), 3),
        "sortino": round(sortino_ratio(returns, 0.0, periods_per_year), 3),
        "max_drawdown": round(max_drawdown(returns), 4),
    }
    if benchmark is not None:
        out["alpha"] = round(alpha(returns, benchmark, periods_per_year), 4)
        out["beta"] = round(beta(returns, benchmark), 3)
    return out
