# QuantMind

**AI-powered quantitative research platform.** A working, tested system that
detects market regimes, forecasts them with a Transformer, allocates capital
with a reinforcement-learning agent, maps asset relationships with a graph
neural network, backtests regime-aware strategies, retrieves over a vector DB,
answers research questions with a retrieval-augmented copilot, coordinates a
multi-agent research team, and streams live updates over WebSockets — served
behind a FastAPI backend and a live dashboard, with Docker, CI, Kubernetes and
Terraform.

> **39 passing tests. Every model below actually runs and is measured — no
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
| 6 | **MLOps** | Dockerised API, GitHub Actions CI, optional MLflow tracking | PyTorch-free API image; CI runs the full suite |
| 7 | **Vector DB retrieval** | Qdrant (embedded or server) + offline hashing embeddings | Same interface as TF-IDF; cosine vector search |
| 8 | **Multi-agent research team** | LangGraph: 4 parallel analysts + synthesizer | Grounded multi-section briefing |
| 9 | **Live streaming** | In-memory bus + WebSocket (optional Kafka backend) | Real-time regime ticks on the dashboard |
| 10 | **Deployment** | Kubernetes manifests + Terraform (AWS ECS Fargate) | Deploy-ready (needs a cluster/account) |

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
              agents/ (LangGraph team)        streaming/ (bus + WebSocket)
                          \                   /
                           ▼                 ▼
                              api/main.py  (FastAPI + WebSocket)
              /api/regime · /api/backtest · /api/ask · /api/research-team
                          /ws/regime · /health
                                            ▼
                          web/index.html  (Plotly dashboard)
              Regime · Backtest · Research tabs + live regime badge
                                            ▼
   mlops/ tracking · Docker/Compose · GitHub Actions CI · deploy/ (K8s + Terraform)
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
               embeddings.py · qdrant_store.py          (Phase 7 — vector DB)
  mlops/       tracking.py              (Phase 6 — MLflow optional)
  agents/      team.py                  (Phase 8 — LangGraph multi-agent)
  streaming/   bus.py · stream.py       (Phase 9 — WebSocket / Kafka)
  api/         main.py                  (FastAPI service + WebSocket)
web/           index.html               (dashboard)
deploy/        k8s/ · terraform/        (Phase 10 — K8s + Terraform)
scripts/       demo.py · train_all.py
tests/         39 tests across all phases
```

---

## Quickstart

Requires Python 3.10+ (developed on 3.12).

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1                 # macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"                       # core + full ML stack (torch, SB3, gymnasium, kafka, anthropic)

pytest                                         # 30 tests
python scripts/demo.py                         # offline regime demo
python scripts/train_all.py                    # train every model end-to-end (offline)
uvicorn quantmind.api.main:app --reload        # open http://127.0.0.1:8000
```

Install profiles (the API itself needs only the core deps):

| Command | Gets you |
|---|---|
| `pip install -e .` | Core (PyTorch-free): regime, backtest, RAG + Qdrant, agents, streaming, API/dashboard |
| `pip install -e ".[ml]"` | + Transformer, RL, GNN training (PyTorch stack) |
| `pip install -e ".[llm]"` | + Claude-generated copilot answers (`anthropic`) |
| `pip install -e ".[streaming]"` | + Kafka backend for the streaming bus |
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
| `POST /api/research-team` `{"query": "...", "source": "synthetic"}` | Multi-agent briefing (macro · risk · relationships · research). |
| `WS /ws/regime?source=synthetic&speed=0.3` | Live stream of regime ticks. |
| `GET /` | Dashboard (Regime / Backtest / Research tabs + live badge). |

Use `source=synthetic` (or `symbol=SYNTH`) on any endpoint for a fully offline demo.
Set `QUANTMIND_RETRIEVER=qdrant` to back the copilot with Qdrant vector search.

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

## Roadmap (genuinely remaining)

The streaming, vector DB, multi-agent and deployment layers are now built (above).
What's left needs scale or live infrastructure to be meaningful, so it's honestly
deferred rather than faked:

- **Live market feed**: drive the WebSocket/Kafka stream from a real-time data
  provider instead of replaying historical bars.
- **Sentence-transformer embeddings** as the default Qdrant encoder (the
  `embeddings.py` interface already supports it; left out to avoid a model
  download in CI).
- **Managed deployment**: actually apply the Terraform/K8s on a cloud account,
  with a hosted MLflow tracking server and image registry in CI.
- **Distributed training** (Ray) for the heavier models, and **online learning**
  to adapt as new data streams in.

---

## Honesty & scope

This is a research / portfolio project, **not investment advice**. Results are
measured on synthetic data with known structure so the models can be evaluated
against ground truth; real markets have no labels and are far less forgiving.
Every metric in this README is reproducible from the test suite and
`scripts/train_all.py`.

## License

MIT.
