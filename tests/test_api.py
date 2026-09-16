from fastapi.testclient import TestClient
import pytest

from quantmind.api.main import app

client = TestClient(app)


def test_health():
    res = client.get("/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert "version" in body


def test_regime_synthetic_offline():
    """The synthetic source must work without any network access."""
    res = client.get("/api/regime", params={"source": "synthetic", "n_states": 3})
    assert res.status_code == 200
    data = res.json()

    assert data["source"] == "synthetic"
    assert data["n_states"] == 3
    assert len(data["series"]) > 100

    first = data["series"][0]
    assert set(first) == {"date", "close", "state", "label", "confidence"}
    assert data["current"]["label"] in {"Bear", "Neutral", "Bull"}
    assert set(data["label_legend"].values()) == {"Bear", "Neutral", "Bull"}


def test_backtest_endpoint_offline():
    res = client.get("/api/backtest", params={"source": "synthetic", "cost_bps": 1.0})
    assert res.status_code == 200
    data = res.json()
    assert "sharpe" in data["strategy"]
    assert "sharpe" in data["benchmark"]
    assert len(data["equity_curve"]) > 100
    assert set(data["equity_curve"][0]) == {"date", "strategy", "benchmark"}


def test_ask_endpoint_offline(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    res = client.post("/api/ask", json={"query": "What is a bear market regime?", "source": "synthetic"})
    assert res.status_code == 200
    data = res.json()
    assert data["used_llm"] is False
    assert data["sources"]
    assert len(data["answer"]) > 0


def test_research_team_endpoint_offline(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    res = client.post("/api/research-team", json={"query": "Summarise the market.", "source": "synthetic"})
    assert res.status_code == 200
    data = res.json()
    assert "Macro" in data["report"]
    assert "current" in data["macro"]
    assert "sharpe" in data["risk"]["strategy"]
    assert data["relationships"]["n_nodes"] > 0


def test_dashboard_served():
    res = client.get("/")
    assert res.status_code == 200
    assert 'id="root"' in res.text
    assert "/assets/" in res.text
    assert "function switchTab" not in res.text


def test_portfolio_diagnostic_endpoint_offline():
    res = client.post("/api/portfolio/diagnose", json={
        "source": "synthetic",
        "holdings": [{"symbol": "SPY", "weight": 0.8}, {"symbol": "TLT", "weight": 0.2}],
    })
    assert res.status_code == 200
    data = res.json()
    assert data["summary"]["effective_holdings"] < 2
    assert any(flag["code"] == "single_holding_concentration" for flag in data["flags"])


def test_portfolio_resilience_endpoint_offline():
    res = client.post("/api/portfolio/resilience", json={
        "source": "synthetic", "simulations": 256, "horizon_days": 20,
        "holdings": [{"symbol": "SPY", "weight": 0.8}, {"symbol": "TLT", "weight": 0.2}],
    })
    assert res.status_code == 200
    data = res.json()
    assert data["simulation"]["simulations"] == 256
    assert len(data["current_allocation"]["risk_attribution"]) == 2
    assert 0 <= data["current_allocation"]["probability_of_loss"] <= 1


def test_forecast_and_relationship_endpoints_offline():
    forecast = client.get("/api/forecast", params={"source": "synthetic", "epochs": 3})
    assert forecast.status_code == 200
    assert sum(forecast.json()["probabilities"].values()) == pytest.approx(1.0, abs=1e-3)
    graph = client.get("/api/relationships")
    assert graph.status_code == 200
    assert graph.json()["graph"]["n_nodes"] > 0


def test_rl_sandbox_endpoint_offline():
    response = client.get("/api/rl-sandbox", params={"source": "synthetic", "timesteps": 256})
    assert response.status_code == 200
    data = response.json()
    assert 0 <= data["average_exposure"] <= 1
    assert "sharpe" in data["strategy"]
