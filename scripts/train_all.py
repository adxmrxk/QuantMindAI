"""Train every model in the platform end-to-end and log results.

    python scripts/train_all.py            # synthetic data (offline)
    python scripts/train_all.py SPY 2y     # live data via yfinance

Set QUANTMIND_MLFLOW=1 (with mlflow installed) to log runs to MLflow;
otherwise metrics are just printed.
"""

from __future__ import annotations

import sys

from quantmind.backtest import backtest_regime_strategy
from quantmind.data import generate_regime_series, load_prices
from quantmind.forecast import train_forecaster
from quantmind.graph import generate_multi_asset_returns, train_sector_gat
from quantmind.mlops import track_run
from quantmind.rl import policy_weights, train_ppo


def main(argv: list[str]) -> None:
    if argv:
        symbol, period = argv[0], (argv[1] if len(argv) > 1 else "2y")
        close = load_prices(symbol, period=period)["close"]
        tag = f"{symbol}_{period}"
    else:
        close = generate_regime_series(n_days=2000, seed=7)["close"]
        tag = "synthetic"

    print(f"\n=== Training QuantMind models on: {tag} ===\n")

    # Phase 1: regime backtest
    with track_run(f"backtest:{tag}", {"phase": 1}) as run:
        bt, _ = backtest_regime_strategy(close)
        run.log_metrics(bt.metrics)
        print(f"[backtest]   sharpe={bt.metrics['sharpe']}  max_dd={bt.metrics['max_drawdown']}")

    # Phase 2: transformer forecaster
    with track_run(f"forecaster:{tag}", {"phase": 2, "seq_len": 30}) as run:
        _, fm = train_forecaster(close, epochs=15)
        run.log_metrics(fm)
        print(f"[forecaster] val_acc={fm['val_accuracy']}  baseline={fm['majority_baseline']}")

    # Phase 3: RL portfolio agent
    with track_run(f"rl_ppo:{tag}", {"phase": 3, "timesteps": 5000}) as run:
        model = train_ppo(close, timesteps=5000)
        weights = policy_weights(model, close)
        rl_metrics = {"avg_weight": round(float(weights.mean()), 4)}
        run.log_metrics(rl_metrics)
        print(f"[rl_ppo]     avg_weight={rl_metrics['avg_weight']}")

    # Phase 4: GNN relationship mapping
    with track_run(f"gat:{tag}", {"phase": 4}) as run:
        returns, sectors = generate_multi_asset_returns(seed=0)
        _, gm = train_sector_gat(returns, sectors, epochs=300)
        run.log_metrics(gm)
        print(f"[gat]        sector_acc={gm['test_accuracy']}")

    print("\nDone. Set QUANTMIND_MLFLOW=1 to persist these runs to MLflow.\n")


if __name__ == "__main__":
    main(sys.argv[1:])
