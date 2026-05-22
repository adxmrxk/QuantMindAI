from fastapi.testclient import TestClient

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
    assert "QuantMind" in res.text
