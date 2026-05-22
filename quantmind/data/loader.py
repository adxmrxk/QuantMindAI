"""Live market data loader.

Fetches daily OHLCV via yfinance and caches it to Parquet so repeat requests
(and offline reruns of an already-fetched symbol) are instant. Returns a
DataFrame indexed by a DatetimeIndex named ``date`` with lower-cased columns.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "cache"

_OHLCV = ["open", "high", "low", "close", "volume"]


def load_prices(
    symbol: str,
    period: str = "2y",
    interval: str = "1d",
    use_cache: bool = True,
) -> pd.DataFrame:
    """Load historical prices for ``symbol``.

    Parameters
    ----------
    symbol: ticker, e.g. ``"SPY"``.
    period: yfinance period string, e.g. ``"1y"``, ``"2y"``, ``"max"``.
    interval: bar size, e.g. ``"1d"``, ``"1h"``.
    use_cache: read/write a local Parquet cache under ``data/cache``.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{symbol.upper()}_{period}_{interval}.parquet"

    if use_cache and cache_path.exists():
        return pd.read_parquet(cache_path)

    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "yfinance is required to fetch live prices. Install it, or call the "
            "API with source='synthetic' for an offline demo."
        ) from exc

    df = yf.download(
        symbol,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False,
    )
    if df is None or df.empty:
        raise ValueError(f"No data returned for symbol {symbol!r} (period={period}).")

    # Newer yfinance can return a column MultiIndex even for a single ticker.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.columns = [str(c).lower() for c in df.columns]
    df.index.name = "date"
    df = df[[c for c in _OHLCV if c in df.columns]].dropna()

    if use_cache:
        df.to_parquet(cache_path)
    return df
