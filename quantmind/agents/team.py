"""The research team: a LangGraph graph of specialist agents.

Topology (fan-out / fan-in)::

        ┌─> macro_analyst ──┐
        ├─> risk_analyst ───┤
  START ┤                   ├─> synthesizer ─> END
        ├─> relationship ───┤
        └─> research_analyst┘

The four analysts write to disjoint state keys, so they run concurrently within
one LangGraph super-step; the synthesizer runs once they have all completed.
"""

from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from quantmind.backtest import backtest_regime_strategy
from quantmind.graph import generate_multi_asset_returns, graph_summary
from quantmind.models import analyze
from quantmind.rag import ResearchCopilot


class ResearchState(TypedDict, total=False):
    query: str
    close: Any
    n_states: int
    macro: dict
    risk: dict
    relationships: dict
    research: dict
    report: str


def macro_analyst(state: ResearchState) -> dict:
    res = analyze(state["close"], n_states=state.get("n_states", 3))
    return {"macro": {"current": res.current, "stats": res.regime_stats()}}


def risk_analyst(state: ResearchState) -> dict:
    bt, _ = backtest_regime_strategy(state["close"], n_states=state.get("n_states", 3))
    return {"risk": {"strategy": bt.metrics, "benchmark": bt.benchmark_metrics}}


def relationship_analyst(state: ResearchState) -> dict:
    returns, _ = generate_multi_asset_returns(seed=0)
    return {"relationships": graph_summary(returns)}


def research_analyst(state: ResearchState) -> dict:
    copilot = ResearchCopilot(close=state["close"])
    return {"research": copilot.ask(state["query"])}


def synthesizer(state: ResearchState) -> dict:
    return {"report": _compose_report(state)}


def _compose_report(state: ResearchState) -> str:
    cur = state["macro"]["current"]
    risk = state["risk"]
    rel = state["relationships"]
    research = state["research"]

    return "\n".join(
        [
            "# QuantMind Research Briefing",
            f"**Question:** {state.get('query', '')}",
            "",
            "## Macro / Regime Analyst",
            f"Current regime is **{cur['label']}** at {cur['confidence'] * 100:.0f}% "
            f"confidence (as of {cur['date']}).",
            "",
            "## Risk Analyst",
            f"Regime-aware Sharpe {risk['strategy']['sharpe']} vs buy-and-hold "
            f"{risk['benchmark']['sharpe']}; max drawdown "
            f"{risk['strategy']['max_drawdown'] * 100:.0f}% vs "
            f"{risk['benchmark']['max_drawdown'] * 100:.0f}%.",
            "",
            "## Relationship Analyst",
            f"Asset universe graph: {rel['n_nodes']} assets, {rel['n_edges']} correlation "
            f"edges, {rel['n_components']} clusters (avg degree {rel['avg_degree']}).",
            "",
            "## Research Analyst",
            research["answer"],
        ]
    )


def build_research_graph():
    """Compile and return the LangGraph research-team app."""
    graph = StateGraph(ResearchState)
    graph.add_node("macro", macro_analyst)
    graph.add_node("risk", risk_analyst)
    graph.add_node("relationship", relationship_analyst)
    graph.add_node("research", research_analyst)
    graph.add_node("synthesizer", synthesizer)

    for analyst in ("macro", "risk", "relationship", "research"):
        graph.add_edge(START, analyst)
        graph.add_edge(analyst, "synthesizer")
    graph.add_edge("synthesizer", END)

    return graph.compile()


def run_research_team(close, query: str, n_states: int = 3) -> dict:
    """Run the full team and return the final state (sections + report)."""
    app = build_research_graph()
    return app.invoke({"query": query, "close": close, "n_states": n_states})
