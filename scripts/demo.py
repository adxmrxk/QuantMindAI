"""Console demo of the regime pipeline — works fully offline.

    python scripts/demo.py            # synthetic data
    python scripts/demo.py SPY 2y     # live data via yfinance
"""

from __future__ import annotations

import sys

from quantmind.data import generate_regime_series, load_prices
from quantmind.models import analyze


def main(argv: list[str]) -> None:
    if argv:
        symbol, period = argv[0], (argv[1] if len(argv) > 1 else "2y")
        print(f"Loading {symbol} ({period}) via yfinance …")
        close = load_prices(symbol, period=period)["close"]
        title = f"{symbol} ({period})"
    else:
        print("No symbol given - using synthetic regime-switching data.")
        close = generate_regime_series()["close"]
        title = "SYNTHETIC"

    result = analyze(close, n_states=3)
    cur = result.current

    print(f"\n=== Regime analysis: {title} ===")
    print(f"Days analysed : {len(result.frame)}")
    print(f"As of         : {cur['date']}")
    print(f"Current regime: {cur['label']}  ({cur['confidence'] * 100:.1f}% confidence)")
    print(f"State legend  : {result.label_legend}\n")

    print(f"{'Regime':<10}{'Days':>6}{'Share':>8}{'AvgRet%':>10}{'AvgVol%':>10}")
    for label, s in sorted(result.regime_stats().items()):
        print(f"{label:<10}{s['days']:>6}{s['share'] * 100:>7.0f}%{s['avg_return']:>10}{s['avg_vol']:>10}")


if __name__ == "__main__":
    main(sys.argv[1:])
