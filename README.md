# QuantMindAI

**A portfolio-risk checkup and quantitative research dashboard for people who
need to understand what their holdings are exposed to.**

QuantMindAI helps a self-directed investor or analyst answer practical,
historical questions before acting: Is one holding dominating the portfolio?
Are the assets actually diversified? What have correlation, drawdown, and daily
loss risk looked like? It is a research tool, not investment advice.

The product is a local FastAPI backend with a React, Vite, and Plotly dashboard.
It runs with synthetic data for a fully reproducible demo, or with
historical market data from yfinance. The dashboard includes regime detection,
a causal walk-forward backtest, a portfolio checkup, grounded research, and a
live WebSocket status feed.

## What you can do

- Enter 2 to 10 holdings and receive normalized weights, concentration flags,
  effective holdings, average correlation, historical drawdown, historical
  95% daily VaR, and a transparent diversification score.
- Run the **resilience lab** to attribute portfolio risk to each holding, model
  2,000 reproducible 20-day paths with a moving-block bootstrap, and compare
  the current allocation with an equal-weight counterfactual.
- Run a **causal walk-forward** regime backtest. The model refits only on data
  that was available at each rebalance, then applies exposure on the following
  day.
- Explore a Transformer regime forecast, correlation graph and GAT sector
  recovery, and a PPO allocation sandbox through documented API routes.
- Ask grounded research questions. It works with an offline extractive answer
  by default and can optionally use Claude when an API key is supplied.

## Measured results

All figures below are reproducible with `python scripts/evaluate_product_metrics.py`.
They use deterministic synthetic data with known structure unless noted.

| Metric | Before | Current | Why it matters |
|---|---:|---:|---|
| Future observations used in displayed backtest | 1,500 | **0** | Eliminates full-history look-ahead from the product backtest |
| Rebalance decisions audited | 0 | **60** | Each causal decision has a trailing-history audit trail |
| Portfolio risk dimensions shown to a user | 0 | **6** | Concentration, effective holdings, correlation, drawdown, VaR, score |
| Transformer validation accuracy | 71.72% baseline | **95.86%** | Synthetic next-regime classification, 290 chronological validation windows |
| GAT sector recovery | 33% random baseline | **100%** | Synthetic 3-sector correlation graph |
| Six-ticker market-data wait | ~720 ms | **~130 ms** | Controlled 120 ms per-ticker I/O benchmark, about 82% lower latency |
| Initial React JavaScript | 4.97 MiB | **156.6 KiB** | Plotly moved to a lazy chart chunk, 96.8% less initial JS |
| Verified automated tests | 39 | **48** | Unit, API, WebSocket, training, portfolio, causal-backtest, resilience, and concurrent-fetch coverage |

The causal backtest is intentionally less flattering than the old full-history
research view: its synthetic Sharpe was 0.364 versus 1.195 for the
full-history result. That drop is evidence that the app no longer presents a
future-informed result as a deployable strategy test.

The live-data latency result is a controlled benchmark, not a promise about a
market-data provider. The React chart engine remains a 4.80 MiB lazy chunk, so
the initial screen is fast while charts retain full Plotly functionality.

## Run it

Python 3.10 or newer is required. The full product routes need the ML extra.
Node.js 20 or newer is required to build the React dashboard.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
cd frontend
npm install
npm run build
cd ..
python -m pytest -q
python -m uvicorn quantmind.api.main:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000 and choose **Demo (offline)** to avoid any market
data dependency. The dashboard starts with the synthetic regime demo.

Docker is optional:

```bash
docker compose up --build
```

Compose starts the API and Qdrant. Docker Desktop must be running first. The
Docker path is configured for the full ML feature set, but it was not verified
on this machine because Docker Desktop was unavailable during the latest run.

For React-only development, keep the FastAPI server running and use
`cd frontend; npm run dev`. Vite proxies API and WebSocket requests to port 8000.

## API

| Route | Purpose |
|---|---|
| `GET /health` | Service health and version |
| `GET /api/regime` | HMM market regime analysis |
| `GET /api/backtest?mode=causal` | Default causal walk-forward backtest |
| `POST /api/portfolio/diagnose` | Portfolio concentration and historical-risk checkup |
| `POST /api/portfolio/resilience` | Risk attribution and reproducible downside simulation |
| `GET /api/forecast` | Transformer next-regime research forecast |
| `GET /api/relationships?train_gat=true` | Correlation graph and optional GAT training |
| `GET /api/rl-sandbox` | PPO allocation research sandbox |
| `POST /api/ask` | Grounded research answer |
| `POST /api/research-team` | LangGraph research briefing |
| `WS /ws/regime` | Regime tick stream |

Use `source=synthetic` for offline API requests. Live prices use yfinance and
need internet access, but no paid market-data account.

## Honest limits

- Historical and synthetic metrics are not proof of future investment returns.
- The Transformer and GAT results are synthetic-model benchmarks. The PPO route
  is explicitly a sandbox, not a trading system.
- Claude generation needs an Anthropic API key and may incur paid API usage.
  The offline RAG answer does not require a key.
- Qdrant embedded mode was verified through the API. The Docker Qdrant service,
  Kubernetes manifests, Terraform plan, cloud deployment, and MLflow backend
  were not run in this verification pass.
- Plotly is bundled with the React chart chunk. The app works without a CDN
  after `npm run build`, although live market history still needs internet access.

Project layout: `quantmind/` contains the backend and models, `web/` contains
the legacy fallback page, `frontend/` contains the React application,
`tests/` contains verification, and `scripts/` contains training and metric
reproduction commands.
