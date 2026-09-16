"""QuantMind API.

Endpoints
---------
GET /health             -> liveness probe
GET /api/regime         -> regime analysis for a symbol (or synthetic demo data)
GET /api/backtest       -> causal walk-forward regime backtest
POST /api/portfolio/diagnose -> portfolio concentration and risk checkup
POST /api/portfolio/resilience -> risk attribution and downside simulation
GET /api/forecast       -> Transformer next-regime research forecast
GET /api/relationships  -> correlation graph and optional GAT research result
GET /api/rl-sandbox     -> PPO allocation research sandbox
GET /                    -> the dashboard (static HTML)

The regime endpoint accepts ``source=synthetic`` (or symbol ``SYNTH``/``DEMO``)
so the whole stack is demoable with no network access.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from quantmind import __version__
from quantmind.agents import run_research_team
from quantmind.backtest import backtest_causal_regime_strategy, backtest_regime_strategy
from quantmind.data import generate_regime_series, load_prices
from quantmind.models import analyze
from quantmind.portfolio import assess_portfolio, assess_resilience
from quantmind.rag import ResearchCopilot
from quantmind.streaming import regime_stream

WEB_DIR = Path(__file__).resolve().parents[2] / "web"
FRONTEND_DIST = Path(__file__).resolve().parents[2] / "frontend" / "dist"

app = FastAPI(
    title="QuantMind API",
    version=__version__,
    description="Portfolio-risk checkup and quantitative market research dashboard.",
)

if (FRONTEND_DIST / "assets").is_dir():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="react-assets")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": __version__}


@lru_cache(maxsize=64)
def _cached_analysis(symbol: str, period: str, n_states: int, source: str):
    """Memoised so repeated dashboard polls don't refit the HMM each time."""
    if source == "synthetic":
        close = generate_regime_series()["close"]
    else:
        close = load_prices(symbol, period=period)["close"]
    return analyze(close, n_states=n_states)


@app.get("/api/regime")
def regime(
    symbol: str = Query("SPY", description="Ticker symbol, e.g. SPY, AAPL."),
    period: str = Query("2y", description="History window, e.g. 1y, 2y, max."),
    n_states: int = Query(3, ge=2, le=5, description="Number of regimes."),
    source: str = Query("auto", description="'auto' (live) or 'synthetic'."),
) -> JSONResponse:
    use_synthetic = source == "synthetic" or symbol.upper() in {"SYNTH", "DEMO"}
    src = "synthetic" if use_synthetic else "auto"

    try:
        result = _cached_analysis(symbol.upper(), period, n_states, src)
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    frame = result.frame
    series = [
        {
            "date": idx.strftime("%Y-%m-%d"),
            "close": round(float(row["close"]), 4),
            "state": int(row["state"]),
            "label": str(row["label"]),
            "confidence": round(float(row["confidence"]), 4),
        }
        for idx, row in frame.iterrows()
    ]

    return JSONResponse(
        {
            "symbol": symbol.upper(),
            "source": src,
            "n_states": n_states,
            "as_of": result.current["date"],
            "current": result.current,
            "label_legend": {str(k): v for k, v in result.label_legend.items()},
            "regime_stats": result.regime_stats(),
            "series": series,
        }
    )


@lru_cache(maxsize=64)
def _cached_backtest(
    symbol: str, period: str, n_states: int, source: str, cost_bps: float, mode: str
):
    if source == "synthetic":
        close = generate_regime_series()["close"]
    else:
        close = load_prices(symbol, period=period)["close"]
    if mode == "causal":
        return (*backtest_causal_regime_strategy(close, n_states=n_states, cost_bps=cost_bps), close)
    if mode == "research":
        result, regime = backtest_regime_strategy(close, n_states=n_states, cost_bps=cost_bps)
        return result, {"mode": "full_history_research", "future_observations_used": len(close)}, close
    raise ValueError("mode must be 'causal' or 'research'")


@app.get("/api/backtest")
def backtest(
    symbol: str = Query("SPY"),
    period: str = Query("2y"),
    n_states: int = Query(3, ge=2, le=5),
    source: str = Query("auto"),
    cost_bps: float = Query(1.0, ge=0.0, le=50.0, description="Round-trip cost (bps)."),
    mode: str = Query("causal", description="'causal' walk-forward or 'research' full-history."),
) -> JSONResponse:
    use_synthetic = source == "synthetic" or symbol.upper() in {"SYNTH", "DEMO"}
    src = "synthetic" if use_synthetic else "auto"

    try:
        bt, audit, _ = _cached_backtest(symbol.upper(), period, n_states, src, cost_bps, mode)
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    curve = [
        {
            "date": idx.strftime("%Y-%m-%d"),
            "strategy": round(float(s), 4),
            "benchmark": round(float(b), 4),
        }
        for idx, s, b in zip(bt.equity.index, bt.equity, bt.benchmark_equity)
    ]
    return JSONResponse(
        {
            "symbol": symbol.upper(),
            "source": src,
            "cost_bps": cost_bps,
            "audit": audit,
            "strategy": bt.metrics,
            "benchmark": bt.benchmark_metrics,
            "equity_curve": curve,
        }
    )


class AskRequest(BaseModel):
    query: str
    symbol: str = "SPY"
    source: str = "synthetic"


class HoldingRequest(BaseModel):
    symbol: str
    weight: float


class PortfolioRequest(BaseModel):
    holdings: list[HoldingRequest]
    period: str = "3y"
    source: str = "synthetic"


class ResilienceRequest(PortfolioRequest):
    simulations: int = Field(default=2000, ge=256, le=10000)
    horizon_days: int = Field(default=20, ge=5, le=60)


@app.post("/api/portfolio/diagnose")
def portfolio_diagnose(req: PortfolioRequest) -> JSONResponse:
    """Historical concentration and drawdown diagnostics for a portfolio."""
    try:
        result = assess_portfolio(
            [holding.model_dump() for holding in req.holdings], period=req.period, source=req.source
        )
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return JSONResponse(result)


@app.post("/api/portfolio/resilience")
def portfolio_resilience(req: ResilienceRequest) -> JSONResponse:
    """Risk attribution plus a reproducible historical downside simulation."""
    try:
        result = assess_resilience(
            [holding.model_dump() for holding in req.holdings],
            period=req.period,
            source=req.source,
            simulations=req.simulations,
            horizon_days=req.horizon_days,
        )
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return JSONResponse(result)


@lru_cache(maxsize=16)
def _cached_copilot(symbol: str, source: str) -> ResearchCopilot:
    if source == "synthetic":
        close = generate_regime_series()["close"]
    else:
        close = load_prices(symbol, period="2y")["close"]
    # Swap in the Qdrant vector backend with QUANTMIND_RETRIEVER=qdrant.
    retriever = None
    if os.getenv("QUANTMIND_RETRIEVER", "tfidf").lower() == "qdrant":
        from quantmind.rag import QdrantRetriever

        retriever = QdrantRetriever()
    return ResearchCopilot(close=close, retriever=retriever)


@app.post("/api/ask")
def ask(req: AskRequest) -> JSONResponse:
    """Natural-language research question, grounded in retrieved documents."""
    use_synthetic = req.source == "synthetic" or req.symbol.upper() in {"SYNTH", "DEMO"}
    src = "synthetic" if use_synthetic else "auto"
    try:
        copilot = _cached_copilot(req.symbol.upper(), src)
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return JSONResponse(copilot.ask(req.query))


def _close_for(symbol: str, source: str):
    if source == "synthetic" or symbol.upper() in {"SYNTH", "DEMO"}:
        return generate_regime_series()["close"], "synthetic"
    return load_prices(symbol.upper(), period="2y")["close"], "auto"


@lru_cache(maxsize=16)
def _cached_forecast(symbol: str, period: str, source: str, epochs: int):
    """Train a compact CPU forecaster once per request key for the demo API."""
    close, resolved_source = _close_for(symbol, source)
    try:
        from quantmind.forecast import train_forecaster
        from quantmind.forecast.dataset import build_dataset
    except ImportError as exc:  # pragma: no cover - depends on optional install
        raise RuntimeError("forecasting needs the ML extra: pip install -e '.[ml]'") from exc
    model, metrics = train_forecaster(close, epochs=epochs)
    windows, _, _ = build_dataset(close)
    probabilities = model.predict_proba(windows[-1])[0]
    labels = analyze(close).label_legend
    return {
        "source": resolved_source,
        "symbol": symbol.upper(),
        "metrics": metrics,
        "next_regime": labels[int(probabilities.argmax())],
        "probabilities": {labels[index]: round(float(value), 4) for index, value in enumerate(probabilities)},
        "warning": "Research forecast only. Regime labels are model-derived, not a trade recommendation.",
    }


@app.get("/api/forecast")
def forecast(
    symbol: str = Query("SPY"),
    period: str = Query("2y"),
    source: str = Query("synthetic"),
    epochs: int = Query(12, ge=3, le=20),
) -> JSONResponse:
    try:
        return JSONResponse(_cached_forecast(symbol, period, source, epochs))
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/api/relationships")
def relationships(train_gat: bool = Query(False)) -> JSONResponse:
    """Expose the correlation-graph feature; GAT training is opt-in and slow."""
    from quantmind.graph import generate_multi_asset_returns, graph_summary

    returns, sectors = generate_multi_asset_returns(seed=0)
    result = {"graph": graph_summary(returns), "assets": int(returns.shape[1]), "days": int(returns.shape[0])}
    if train_gat:
        try:
            from quantmind.graph import train_sector_gat
        except ImportError as exc:  # pragma: no cover - optional dependency path
            raise HTTPException(status_code=503, detail="GAT needs the ML extra") from exc
        _, result["gat"] = train_sector_gat(returns, sectors, epochs=300, seed=0)
    return JSONResponse(result)


@lru_cache(maxsize=8)
def _cached_rl_sandbox(symbol: str, source: str, timesteps: int):
    """Train a small PPO allocation policy for an explicitly research-only run."""
    close, resolved_source = _close_for(symbol, source)
    try:
        from quantmind.rl import policy_weights, train_ppo
    except ImportError as exc:  # pragma: no cover - optional dependency path
        raise RuntimeError("PPO needs the ML extra: pip install -e '.[ml]'") from exc
    from quantmind.backtest import run_backtest

    policy = train_ppo(close, timesteps=timesteps, seed=0)
    weights = policy_weights(policy, close)
    result = run_backtest(close.loc[weights.index], weights, cost_bps=1.0)
    return {
        "source": resolved_source,
        "symbol": symbol.upper(),
        "timesteps": timesteps,
        "average_exposure": round(float(weights.mean()), 4),
        "strategy": result.metrics,
        "benchmark": result.benchmark_metrics,
        "warning": "Sandbox result on the selected history only. It is not investment advice or a deployable policy.",
    }


@app.get("/api/rl-sandbox")
def rl_sandbox(
    symbol: str = Query("SPY"),
    source: str = Query("synthetic"),
    timesteps: int = Query(1200, ge=256, le=5000),
) -> JSONResponse:
    try:
        return JSONResponse(_cached_rl_sandbox(symbol, source, timesteps))
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/api/research-team")
def research_team(req: AskRequest) -> JSONResponse:
    """Run the multi-agent research team and return its briefing + sections."""
    try:
        close, _ = _close_for(req.symbol, req.source)
        state = run_research_team(close, req.query)
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return JSONResponse(
        {
            "report": state["report"],
            "macro": state["macro"],
            "risk": state["risk"],
            "relationships": state["relationships"],
            "research_sources": state["research"]["sources"],
        }
    )


@app.websocket("/ws/regime")
async def ws_regime(websocket: WebSocket) -> None:
    """Stream live regime updates to the dashboard over a WebSocket."""
    await websocket.accept()
    p = websocket.query_params
    symbol, source = p.get("symbol", "SPY"), p.get("source", "synthetic")
    limit, speed = int(p.get("limit", "150")), float(p.get("speed", "0.3"))
    try:
        close, _ = _close_for(symbol, source)
        async for update in regime_stream(close, limit=limit, speed=speed):
            await websocket.send_json(update)
        await websocket.send_json({"type": "complete"})
    except WebSocketDisconnect:
        return
    except (ValueError, RuntimeError) as exc:
        await websocket.send_json({"type": "error", "detail": str(exc)})
    finally:
        try:
            await websocket.close()
        except RuntimeError:
            pass


@app.get("/")
def dashboard() -> FileResponse:
    """Serve the compiled React dashboard, with the legacy static page as fallback."""
    index = FRONTEND_DIST / "index.html"
    return FileResponse(index if index.is_file() else WEB_DIR / "index.html")
