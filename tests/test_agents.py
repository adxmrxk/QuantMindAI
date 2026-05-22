from quantmind.agents import build_research_graph, run_research_team
from quantmind.data import generate_regime_series


def test_graph_compiles():
    app = build_research_graph()
    assert app is not None


def test_research_team_produces_full_briefing(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    close = generate_regime_series(n_days=500, seed=7)["close"]
    state = run_research_team(close, "Why might the regime strategy beat buy and hold?")

    # Every analyst contributed.
    for key in ("macro", "risk", "relationships", "research"):
        assert key in state, f"missing analyst output: {key}"

    # The synthesizer produced a report touching each section.
    report = state["report"]
    assert "Macro" in report
    assert "Risk Analyst" in report
    assert "Relationship Analyst" in report
    assert "Research Analyst" in report
    assert state["macro"]["current"]["label"] in {"Bull", "Bear", "Neutral"}
