# QuantMind

**AI-powered quantitative research platform.** A working, tested system that
detects market regimes, forecasts them with a Transformer, allocates capital
with a reinforcement-learning agent, maps asset relationships with a graph
neural network, backtests regime-aware strategies, and answers research
questions with a retrieval-augmented copilot — served behind a FastAPI backend
and a live dashboard, with Docker and CI.

> **30 passing tests. Every model below actually runs and is measured — no
> stubs.** The synthetic data generator means the whole platform also runs
> offline, and gives ground-truth labels to evaluate the models against.

---

## What's built

| # | Capability | Approach | Headline result\* |
|---|---|---|---|
| 0 | **Regime detection** | Gaussian HMM (multi-restart) over momentum + volatility | 85.7% regime recovery; persistent states (27 switches / 1480 days) |
| 1 | **Backtesting engine** | Regime-aware allocation vs buy & hold; Sharpe/Sortino/drawdown/alpha/beta | Sharpe **1.20 vs 0.39**, max drawdown **−19% vs −48%** |
| 2 | **Transformer forecaster** | PyTorch Transformer encoder predicting next-day regime | **96.9%** val accuracy vs 47.7% majority baseline |
| 3 | **RL portfolio agent** | Gymnasium env + PPO (Stable-Baselines3), risk-penalised reward | Trains end-to-end; policy is backtestable |
| 4 | **GNN relationship mapping** | Correlation graph + from-scratch Graph Attention Network | **100%** sector recovery; graph splits into the right components |
| 5 | **RAG research copilot** | TF-IDF retrieval over live + static docs, optional Claude generation | Grounded answers citing sources; works offline |
| 6 | **MLOps** | Dockerised API, GitHub Actions CI, optional MLflow tracking | Lean API image; CI runs the full suite |

\* Measured on synthetic data with known structure (seed-fixed, reproducible).

---

## Architecture

```
                       ┌────────────────────────────────────────────┐
   yfinance / synthetic│                  DATA                       │
        prices ───────►│   loader.py (Parquet cache) · synthetic.py  │
                       └───────────────┬────────────────────────────┘
                                       ▼  features (ret · momentum · volatility)
   ┌───────────────┬───────────────┬───┴───────────┬───────────────┬──────────────┐
   ▼               ▼               ▼               ▼               ▼              ▼
 models/         forecast/        rl/            graph/          backtest/       rag/
 Gaussian HMM    Transformer      PPO agent      GAT over        regime          retrieval +
 regime label    next-day         (Gymnasium)    correlation     strategy vs     optional
 (smoother)      regime forecast  allocation     graph           benchmark       Claude
   └───────────────┴───────────────┴───────┬───────┴───────────────┴──────────────┘
                                            ▼
                              api/main.py  (FastAPI)
                    /api/regime · /api/backtest · /api/ask · /health
                                            ▼
                          web/index.html  (Plotly dashboard)
                     Regime · Backtest · Research Copilot tabs
                                            ▼
              mlops/ tracking · Dockerfile · docker-compose · GitHub Actions CI
```

```
quantmind/
  data/        loader.py · synthetic.py
  features/    indicators.py            (momentum + volatility)
  models/      regime.py                (Phase 0 — Gaussian HMM)
  backtest/    metrics.py · strategies.py · engine.py   (Phase 1)
  forecast/    transformer.py · dataset.py · train.py   (Phase 2)
  rl/          env.py · agent.py        (Phase 3 — Gymnasium + PPO)
  graph/       build.py · gat.py · train.py             (Phase 4 — GAT)
  rag/         retriever.py · corpus.py · copilot.py    (Phase 5)
  mlops/       tracking.py              (Phase 6 — MLflow optional)
  api/         main.py                  (FastAPI service)
web/           index.html               (dashboard)
scripts/       demo.py · train_all.py
tests/         28 model/engine/API tests + mlops  (30 total)
```

---

## Quickstart

Requires Python 3.10+ (developed on 3.12).

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1                 # macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"                       # core + ML stack (torch, SB3, gymnasium, networkx, anthropic)

pytest                                         # 30 tests
python scripts/demo.py                         # offline regime demo
python scripts/train_all.py                    # train every model end-to-end (offline)
uvicorn quantmind.api.main:app --reload        # open http://127.0.0.1:8000
```

Install profiles (the API itself needs only the core deps):

| Command | Gets you |
|---|---|
| `pip install -e .` | Core: regime, backtest, RAG retrieval, API/dashboard |
| `pip install -e ".[ml]"` | + Transformer, RL, GNN (PyTorch stack) |
| `pip install -e ".[llm]"` | + Claude-generated copilot answers (`anthropic`) |
| `pip install -e ".[dev]"` | Everything above + test tooling |

### Run with Docker

```bash
docker compose up --build      # then open http://localhost:8000
```

### API

| Endpoint | Description |
|---|---|
| `GET /health` | Liveness probe. |
| `GET /api/regime?symbol=SPY&period=2y&n_states=3&source=auto` | Regime classification + per-regime stats. |
| `GET /api/backtest?symbol=SPY&source=auto&cost_bps=1` | Regime strategy vs buy & hold: equity curves + metrics. |
| `POST /api/ask` `{"query": "...", "source": "synthetic"}` | Grounded research answer with sources. |
| `GET /` | Dashboard (Regime / Backtest / Research tabs). |

Use `source=synthetic` (or `symbol=SYNTH`) on any endpoint for a fully offline demo.

### Claude-powered copilot (optional)

```powershell
$env:ANTHROPIC_API_KEY = "sk-ant-..."   # then POST /api/ask uses Claude (with prompt caching)
```
Without a key the copilot returns a grounded **extractive** answer, so it always works.

---

## Modeling notes (the interesting bits)

- **Regimes need smoothed features.** Feeding the HMM raw daily returns fails —
  daily drift (~0.1%) is dwarfed by volatility (~1–2%), so the states are
  inseparable and flicker. Momentum + volatility, plus **multi-restart fitting**
  to dodge degenerate local optima, are what make the regimes stick.
- **The forecaster's 97% is honest but flattered by persistence** — regimes
  rarely change day to day, so next-day prediction is easy. The same Transformer
  applied to next-day *direction* is the harder, natural extension.
- **The GNN earns its keep from structure.** Node features alone are weak for
  sector ID; the correlation graph carries the signal, and the GAT exploits it
  (100% sector recovery, with the graph splitting into the right components).
- **The copilot never free-associates** — it answers only from retrieved
  documents (including live, data-grounded ones) and returns its sources.

---

## Roadmap (not yet built)

These need external infrastructure or accounts to run, so they're deliberately
left as the next layer rather than faked:

- **Live streaming** (Kafka/WebSockets) for real-time updates.
- **Vector DB upgrade**: swap TF-IDF for sentence-transformer embeddings in
  Qdrant (the retriever interface already matches).
- **Multi-agent orchestration** (LangGraph) coordinating specialist agents.
- **Cloud deployment** (Kubernetes/Terraform on AWS/GCP) and live MLflow server.
- **Distributed training** (Ray) for the heavier models.

---

## Honesty & scope

This is a research / portfolio project, **not investment advice**. Results are
measured on synthetic data with known structure so the models can be evaluated
against ground truth; real markets have no labels and are far less forgiving.
Every metric in this README is reproducible from the test suite and
`scripts/train_all.py`.

## License

MIT.
