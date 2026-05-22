"""QuantMind API.

Endpoints
---------
GET /health             -> liveness probe
GET /api/regime         -> regime analysis for a symbol (or synthetic demo data)
GET /                    -> the dashboard (static HTML)

The regime endpoint accepts ``source=synthetic`` (or symbol ``SYNTH``/``DEMO``)
so the whole stack is demoable with no network access.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from quantmind import __version__
from quantmind.backtest import backtest_regime_strategy
from quantmind.data import generate_regime_series, load_prices
from quantmind.models import analyze
from quantmind.rag import ResearchCopilot

WEB_DIR = Path(__file__).resolve().parents[2] / "web"

app = FastAPI(
    title="QuantMind API",
    version=__version__,
    description="Market regime detection — the first slice of the QuantMind platform.",
)


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
def _cached_backtest(symbol: str, period: str, n_states: int, source: str, cost_bps: float):
    if source == "synthetic":
        close = generate_regime_series()["close"]
    else:
        close = load_prices(symbol, period=period)["close"]
    return backtest_regime_strategy(close, n_states=n_states, cost_bps=cost_bps)


@app.get("/api/backtest")
def backtest(
    symbol: str = Query("SPY"),
    period: str = Query("2y"),
    n_states: int = Query(3, ge=2, le=5),
    source: str = Query("auto"),
    cost_bps: float = Query(1.0, ge=0.0, le=50.0, description="Round-trip cost (bps)."),
) -> JSONResponse:
    use_synthetic = source == "synthetic" or symbol.upper() in {"SYNTH", "DEMO"}
    src = "synthetic" if use_synthetic else "auto"

    try:
        bt, _ = _cached_backtest(symbol.upper(), period, n_states, src, cost_bps)
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
            "strategy": bt.metrics,
            "benchmark": bt.benchmark_metrics,
            "equity_curve": curve,
        }
    )


class AskRequest(BaseModel):
    query: str
    symbol: str = "SPY"
    source: str = "synthetic"


@lru_cache(maxsize=16)
def _cached_copilot(symbol: str, source: str) -> ResearchCopilot:
    if source == "synthetic":
        close = generate_regime_series()["close"]
    else:
        close = load_prices(symbol, period="2y")["close"]
    return ResearchCopilot(close=close)


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


@app.get("/")
def dashboard() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")
