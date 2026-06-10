# QuantMindAI

**AI-powered quantitative research platform.**

QuantMindAI is a single working system that detects market regimes with a Gaussian HMM, forecasts the next regime with a Transformer, allocates capital with a reinforcement-learning agent, maps asset relationships with a graph attention network, backtests regime-aware strategies, retrieves over a vector database, answers research questions with a grounded retrieval-augmented copilot, coordinates a multi-agent research team, and streams live updates over WebSockets. The whole stack is served behind a FastAPI backend and a Plotly dashboard, with Docker, CI, Kubernetes manifests, and Terraform for deployment.

> **39 passing tests. Every model below actually runs and is measured — no stubs.** A deterministic synthetic data generator lets the whole platform run offline, and gives ground-truth labels to evaluate the models against.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Capability Matrix](#capability-matrix)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [API](#api)
- [Modeling Notes](#modeling-notes)
- [Deployment](#deployment)
- [Honesty and Scope](#honesty-and-scope)

---

## Project Overview

### What It Is

A vertically integrated research platform that combines classical statistical modeling (HMM regimes, correlation graphs), modern deep learning (Transformer forecaster, Graph Attention Network), reinforcement learning (PPO policy on a Gymnasium environment), retrieval-augmented generation (TF-IDF or Qdrant, optional Claude), and multi-agent orchestration (LangGraph) into one composable Python package. The same models that run in the test suite are the ones the FastAPI service serves to the dashboard.

Every capability is gated behind an extras install so the API container can ship without PyTorch (no torch in the runtime image unless you ask for it).

### What It Does

- Detects market regimes from price data using a Gaussian HMM with multi-restart fitting
- Forecasts next-day regimes with a PyTorch Transformer encoder
- Trains a PPO portfolio agent in a custom Gymnasium environment with risk-penalized rewards
- Builds correlation graphs across asset universes and trains a from-scratch Graph Attention Network on them
- Backtests regime-aware allocation strategies against buy-and-hold benchmarks with full risk metrics (Sharpe, Sortino, drawdown, alpha, beta)
- Retrieves relevant documents over TF-IDF or Qdrant for a grounded research copilot
- Coordinates a four-analyst LangGraph research team that produces a multi-section briefing
- Streams live regime ticks to the dashboard over WebSockets (with optional Kafka backend)
- Exposes everything through a FastAPI service with a Plotly dashboard

---

## Capability Matrix

Each row is implemented, tested, and measured. Headline results are on synthetic data with known structure (seed-fixed, reproducible).

| # | Capability | Approach | Headline result |
|---|---|---|---|
| 0 | **Regime detection** | Gaussian HMM (multi-restart) over momentum + volatility features | 85.7% regime recovery; persistent states (27 switches per 1480 days) |
| 1 | **Backtesting engine** | Regime-aware allocation vs buy-and-hold; Sharpe, Sortino, drawdown, alpha, beta | Sharpe **1.20 vs 0.39**, max drawdown **−19% vs −48%** |
| 2 | **Transformer forecaster** | PyTorch Transformer encoder predicting next-day regime | **96.9%** validation accuracy vs 47.7% majority baseline |
| 3 | **RL portfolio agent** | Gymnasium env + PPO via Stable-Baselines3, risk-penalized reward | Trains end-to-end, policy is directly backtestable |
| 4 | **GNN relationship mapping** | Correlation graph + from-scratch Graph Attention Network | **100%** sector recovery; graph splits into the right components |
| 5 | **RAG research copilot** | TF-IDF retrieval over live + static docs, optional Claude generation | Grounded answers citing sources, works offline without an API key |
| 6 | **MLOps** | Dockerized API, GitHub Actions CI, optional MLflow tracking | PyTorch-free API image, CI runs the full suite |
| 7 | **Vector DB retrieval** | Qdrant (embedded or server) with offline hashing embeddings | Same interface as TF-IDF, cosine vector search |
| 8 | **Multi-agent research team** | LangGraph: 4 parallel analysts + synthesizer | Grounded multi-section briefing |
| 9 | **Live streaming** | In-memory bus + WebSocket (optional Kafka backend) | Real-time regime ticks on the dashboard |
| 10 | **Deployment** | Kubernetes manifests + Terraform (AWS ECS Fargate) | Deploy-ready (needs a cluster or account) |

---

## Architecture

```
                       ┌────────────────────────────────────────────┐
   yfinance / synthetic│                  DATA                       │
        prices ───────▶│   loader.py (Parquet cache)  ·  synthetic.py│
                       └───────────────┬────────────────────────────┘
                                       ▼  features (return + momentum + volatility)
   ┌───────────────┬───────────────┬───┴───────────┬───────────────┬──────────────┐
   ▼               ▼               ▼               ▼               ▼              ▼
 models/         forecast/        rl/            graph/          backtest/       rag/
 Gaussian HMM    Transformer      PPO agent      GAT over        regime          retrieval +
 regime label    next-day         (Gymnasium)    correlation     strategy vs     optional
 (smoother)      regime forecast  allocation     graph           benchmark       Claude
   └───────────────┴───────────────┴───────┬───────┴───────────────┴──────────────┘
                                            ▼
              agents/ (LangGraph team)             streaming/ (bus + WebSocket)
                          ╲                         ╱
                           ▼                       ▼
                          api/main.py  (FastAPI + WebSocket)
              /api/regime · /api/backtest · /api/ask · /api/research-team
                          /ws/regime · /health
                                            ▼
                          web/index.html  (Plotly dashboard)
              Regime  ·  Backtest  ·  Research tabs + live regime badge
                                            ▼
   mlops/ tracking · Docker / Compose · GitHub Actions CI · deploy/ (K8s + Terraform)
```

---

## Tech Stack

### Core ML

| Component                       | Purpose |
|---------------------------------|---------|
| **Gaussian HMM** (`hmmlearn`)   | Regime detection over momentum and volatility features. Multi-restart fitting to dodge degenerate local optima. |
| **PyTorch Transformer encoder** | Next-day regime forecasting from a windowed feature history. |
| **Stable-Baselines3 PPO**       | RL portfolio allocation agent over a custom Gymnasium environment. |
| **Custom Graph Attention Network** | From-scratch GAT layers trained on asset correlation graphs for sector recovery. |

### Retrieval and Agents

| Component             | Purpose |
|-----------------------|---------|
| **TF-IDF retriever**  | Default, offline, zero-dependency document retriever. |
| **Qdrant**            | Optional vector backend (embedded or server) selected by `QUANTMIND_RETRIEVER=qdrant`. |
| **Anthropic Claude**  | Optional generation layer in the copilot; the answer always falls back to grounded extractive mode without an API key. |
| **LangGraph**         | Multi-agent orchestration. Four analysts (macro, risk, relationship, research) fan out in parallel, synthesizer fans in. |

### Serving

| Component         | Purpose |
|-------------------|---------|
| **FastAPI + Uvicorn** | REST endpoints and a `/ws/regime` WebSocket for live streaming. |
| **Plotly + vanilla JS** | Single-page dashboard (`web/index.html`) with Regime, Backtest, and Research tabs plus a live regime badge. |
| **`lru_cache`**   | Memoizes HMM fits and backtests so repeated dashboard polls do not refit. |

### MLOps and Deployment

| Component             | Purpose |
|-----------------------|---------|
| **Docker + Compose**  | Local stack startup. API image is PyTorch-free by default. |
| **GitHub Actions CI** | Runs the full test suite on every push. |
| **MLflow (optional)** | Experiment and run tracking via `quantmind.mlops.tracking`. |
| **Kubernetes**        | Deployment, Service, and HPA manifests under `deploy/k8s/`. |
| **Terraform**         | AWS ECS Fargate provisioning under `deploy/terraform/`. |

### Data

| Component                      | Purpose |
|--------------------------------|---------|
| **yfinance**                   | Live price loader with Parquet cache. |
| **Synthetic regime generator** | Deterministic price series with known regime labels; makes the whole platform usable offline and gives ground truth for evaluation. |

---

## Project Structure

```
quantmindai/
│
├── quantmind/                      Importable library
│   ├── data/
│   │   ├── loader.py               yfinance + Parquet cache
│   │   └── synthetic.py            Regime-labeled synthetic series
│   ├── features/
│   │   └── indicators.py           Momentum + volatility features
│   ├── models/
│   │   └── regime.py               Phase 0: Gaussian HMM + analyzer
│   ├── backtest/
│   │   ├── strategies.py           Regime-aware allocation
│   │   ├── engine.py               Backtester
│   │   └── metrics.py              Sharpe, Sortino, drawdown, alpha, beta
│   ├── forecast/
│   │   ├── dataset.py              Sliding-window dataset
│   │   ├── transformer.py          Transformer encoder
│   │   └── train.py                Training loop
│   ├── rl/
│   │   ├── env.py                  Gymnasium environment
│   │   └── agent.py                PPO agent
│   ├── graph/
│   │   ├── build.py                Correlation graph construction
│   │   ├── gat.py                  From-scratch Graph Attention Network
│   │   └── train.py                Sector-recovery training loop
│   ├── rag/
│   │   ├── retriever.py            TF-IDF retriever
│   │   ├── corpus.py               Document loaders
│   │   ├── copilot.py              Grounded answer composition
│   │   ├── embeddings.py           Offline hashing embeddings (Qdrant default)
│   │   └── qdrant_store.py         Qdrant retriever wrapper
│   ├── mlops/
│   │   └── tracking.py             Optional MLflow integration
│   ├── agents/
│   │   └── team.py                 LangGraph multi-agent research team
│   ├── streaming/
│   │   ├── bus.py                  In-memory pub/sub (+ optional Kafka)
│   │   └── stream.py               Async regime stream generator
│   └── api/
│       └── main.py                 FastAPI app + WebSocket
│
├── web/
│   └── index.html                  Plotly dashboard
│
├── deploy/
│   ├── k8s/
│   │   ├── deployment.yaml
│   │   ├── service.yaml
│   │   └── hpa.yaml
│   └── terraform/
│       ├── main.tf
│       ├── variables.tf
│       └── outputs.tf
│
├── scripts/
│   ├── demo.py                     Offline regime demo
│   └── train_all.py                End-to-end training of every model
│
├── tests/                          39 tests across all 11 capabilities
├── data/cache/                     Parquet price cache
│
├── pyproject.toml                  Core + ml + llm + streaming + dev extras
├── docker-compose.yml
├── Dockerfile
└── README.md
```

---

## Getting Started

Requires Python 3.10+ (developed on 3.12).

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1                 # macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"                       # core + full ML stack

pytest                                         # 39 tests
python scripts/demo.py                         # offline regime demo
python scripts/train_all.py                    # train every model end-to-end (offline)
uvicorn quantmind.api.main:app --reload        # open http://127.0.0.1:8000
```

### Install profiles

The API itself only needs the core deps.

| Command                              | Adds |
|--------------------------------------|------|
| `pip install -e .`                   | Core (PyTorch-free): regime, backtest, RAG + Qdrant, agents, streaming, API, dashboard |
| `pip install -e ".[ml]"`             | Transformer, RL, GNN training (PyTorch stack) |
| `pip install -e ".[llm]"`            | Claude-generated copilot answers (`anthropic`) |
| `pip install -e ".[streaming]"`      | Kafka backend for the streaming bus |
| `pip install -e ".[dev]"`            | Everything above plus test tooling |

### Run with Docker

```bash
docker compose up --build      # then open http://localhost:8000
```

### Claude-powered copilot (optional)

```powershell
$env:ANTHROPIC_API_KEY = "sk-ant-..."   # POST /api/ask now uses Claude with prompt caching
```

Without a key the copilot returns a grounded **extractive** answer, so it always works.

---

## API

| Endpoint | Description |
|---|---|
| `GET /health` | Liveness probe |
| `GET /api/regime?symbol=SPY&period=2y&n_states=3&source=auto` | Regime classification + per-regime stats |
| `GET /api/backtest?symbol=SPY&source=auto&cost_bps=1` | Regime strategy vs buy-and-hold: equity curves + metrics |
| `POST /api/ask` body `{"query": "...", "source": "synthetic"}` | Grounded research answer with sources |
| `POST /api/research-team` body `{"query": "...", "source": "synthetic"}` | Multi-agent briefing (macro, risk, relationships, research) |
| `WS /ws/regime?source=synthetic&speed=0.3` | Live stream of regime ticks |
| `GET /` | Dashboard (Regime / Backtest / Research tabs + live badge) |

Use `source=synthetic` (or `symbol=SYNTH`) on any endpoint for a fully offline demo.

---

## Modeling Notes

The lessons that fell out of building this.

- **Regimes need smoothed features.** Feeding the HMM raw daily returns fails. Daily drift (~0.1%) is dwarfed by volatility (~1–2%), so the states are inseparable and flicker. Momentum and volatility, plus multi-restart fitting to dodge degenerate local optima, are what make the regimes stick.
- **The forecaster's 97% is honest but flattered by persistence.** Regimes rarely change day to day, so next-day prediction is easy. The same Transformer applied to next-day *direction* is the harder, natural extension.
- **The GNN earns its keep from structure.** Node features alone are weak for sector identification. The correlation graph carries the signal, and the GAT exploits it (100% sector recovery, with the graph splitting into the right components).
- **The copilot never free-associates.** It answers only from retrieved documents (including live, data-grounded ones) and returns its sources.

---

## Deployment

Both deployment paths are provisioned in code and ready to apply.

**Kubernetes** ([deploy/k8s/](deploy/k8s/)): Deployment, ClusterIP Service, and HorizontalPodAutoscaler manifests. Apply with `kubectl apply -k deploy/k8s/` after pushing an image to a registry the cluster can pull from.

**Terraform on AWS** ([deploy/terraform/](deploy/terraform/)): provisions an ECS Fargate service with the necessary IAM, networking, and load-balancing resources. Run `terraform init && terraform apply` after editing `terraform.tfvars`.

Neither path has been applied to a live cluster or account in this repo — that requires credentials, an image push to a registry, and a hosted MLflow tracking endpoint, which are deliberately deferred. The manifests render cleanly (`kubectl kustomize`) and the Terraform validates (`terraform validate`).

---

## Honesty and Scope

This is a research and portfolio project, **not investment advice**. Results are measured on synthetic data with known structure so the models can be evaluated against ground truth; real markets have no labels and are far less forgiving. Every metric in this README is reproducible from the test suite and `scripts/train_all.py`.

License: MIT.
