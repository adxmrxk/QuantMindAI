"""Multi-agent research orchestration with LangGraph.

A LangGraph ``StateGraph`` coordinates four specialist analyst nodes that run in
parallel — macro/regime, risk, relationships (graph), and research (RAG) — then a
synthesizer node fuses their findings into a single briefing. Each agent calls
the platform's own components, so the team's output is grounded in real model
results rather than free-form text.
"""

from quantmind.agents.team import build_research_graph, run_research_team

__all__ = ["build_research_graph", "run_research_team"]
