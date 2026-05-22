"""The document corpus the copilot retrieves over.

Combines static market-knowledge snippets with *live* documents generated from
the platform's own outputs, so answers are grounded in the same numbers the
dashboard shows.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Document:
    id: str
    title: str
    text: str


STATIC_DOCS: list[Document] = [
    Document(
        "regimes",
        "Market regimes",
        "A market regime is a persistent state of return and risk. QuantMind tracks "
        "three: Bull (positive trend, low volatility), Bear (negative trend, high "
        "volatility), and Neutral (sideways, moderate volatility). Regimes are "
        "detected with a Gaussian Hidden Markov Model over momentum and volatility.",
    ),
    Document(
        "volatility",
        "Volatility and bear markets",
        "Volatility tends to spike during market declines: bear regimes are "
        "characterised by both negative drift and elevated volatility. A jump in "
        "realised volatility is often the earliest signal of a regime shift away "
        "from a calm bull market.",
    ),
    Document(
        "sharpe",
        "Performance metrics",
        "The Sharpe ratio is annualised return divided by annualised volatility; "
        "the Sortino ratio penalises only downside volatility. Maximum drawdown is "
        "the largest peak-to-trough equity decline. A higher Sharpe with a smaller "
        "drawdown indicates better risk-adjusted performance.",
    ),
    Document(
        "sectors",
        "Sector relationships",
        "Stocks in the same sector are driven by a common sector factor and so move "
        "together; their returns are highly correlated. QuantMind models the asset "
        "universe as a correlation graph and uses a graph attention network to map "
        "these relationships and detect contagion between related assets.",
    ),
    Document(
        "regime_strategy",
        "Regime-aware allocation",
        "A regime-aware strategy reduces or eliminates equity exposure during Bear "
        "regimes and stays invested during Bull regimes. By sitting out the "
        "high-volatility bear periods it typically lowers maximum drawdown and "
        "raises the Sharpe ratio relative to buy-and-hold.",
    ),
]


def build_market_corpus(close: pd.Series | None = None, n_states: int = 3) -> list[Document]:
    """Static knowledge plus live, data-grounded documents when ``close`` is given."""
    docs = list(STATIC_DOCS)
    if close is None:
        return docs

    from quantmind.backtest import backtest_regime_strategy
    from quantmind.models import analyze

    regime = analyze(close, n_states=n_states)
    cur = regime.current
    stats = regime.regime_stats()
    stat_text = " ".join(
        f"The {label} regime covered {s['days']} days ({s['share'] * 100:.0f}% of "
        f"history) with average daily return {s['avg_return']}% and volatility "
        f"{s['avg_vol']}%."
        for label, s in stats.items()
    )
    docs.append(
        Document(
            "live_regime",
            "Current regime snapshot",
            f"As of {cur['date']} the market is in a {cur['label']} regime with "
            f"{cur['confidence'] * 100:.0f}% confidence. {stat_text}",
        )
    )

    bt, _ = backtest_regime_strategy(close, n_states=n_states)
    docs.append(
        Document(
            "live_backtest",
            "Regime strategy backtest",
            f"The regime-aware strategy achieved a Sharpe ratio of {bt.metrics['sharpe']}, "
            f"total return {bt.metrics['total_return'] * 100:.0f}%, and maximum drawdown "
            f"{bt.metrics['max_drawdown'] * 100:.0f}%. Buy-and-hold returned a Sharpe of "
            f"{bt.benchmark_metrics['sharpe']} with maximum drawdown "
            f"{bt.benchmark_metrics['max_drawdown'] * 100:.0f}%.",
        )
    )
    return docs
