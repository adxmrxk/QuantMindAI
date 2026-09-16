import { lazy, Suspense, useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";

const Plot = lazy(() => import("react-plotly.js"));
const COLORS = { Bull: "#2ecc71", Bear: "#e74c3c", Neutral: "#f1c40f" };
const pctMetrics = new Set(["total_return", "cagr", "ann_volatility", "max_drawdown", "alpha"]);
const backtestMetrics = ["total_return", "cagr", "sharpe", "sortino", "ann_volatility", "max_drawdown", "alpha", "beta"];

async function api(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail || `HTTP ${response.status}`);
  }
  return response.json();
}

function Panel({ title, children, className = "" }) {
  return <section className={`panel ${className}`}><h2>{title}</h2>{children}</section>;
}

function Chart(props) {
  return <Suspense fallback={<div className="meta">Getting the chart ready...</div>}><Plot {...props} /></Suspense>;
}

function App() {
  const [tab, setTab] = useState("regime");
  const [symbol, setSymbol] = useState("SPY");
  const [period, setPeriod] = useState("2y");
  const [states, setStates] = useState(3);
  const [cost, setCost] = useState(1);
  const [status, setStatus] = useState("Getting the demo ready...");
  const [live, setLive] = useState("Connecting...");
  const [regime, setRegime] = useState(null);
  const [backtest, setBacktest] = useState(null);
  const [holdings, setHoldings] = useState("SPY:60, TLT:40");
  const [portfolio, setPortfolio] = useState(null);
  const [resilience, setResilience] = useState(null);
  const [question, setQuestion] = useState("");
  const [research, setResearch] = useState({ answer: "Ask about the market, a regime, or how the strategy behaved.", sources: [], note: "" });

  const parsedHoldings = () => holdings.split(",").map((part) => {
    const [ticker, weight] = part.trim().split(":");
    return { symbol: ticker, weight: Number(weight) };
  });

  const runRegime = async (source) => {
    setStatus(`Looking at ${source === "synthetic" ? "the sample market" : symbol}...`);
    const data = await api(`/api/regime?symbol=${encodeURIComponent(symbol || "SPY")}&period=${period}&n_states=${states}&source=${source}`);
    setRegime(data);
    setStatus(`${source === "synthetic" ? "Sample market" : symbol} is ready to explore.`);
  };

  const runBacktest = async (source) => {
    setStatus(`Testing the strategy against buy and hold for ${source === "synthetic" ? "the sample market" : symbol}...`);
    const data = await api(`/api/backtest?symbol=${encodeURIComponent(symbol || "SPY")}&period=${period}&n_states=${states}&source=${source}&cost_bps=${cost}&mode=causal`);
    setBacktest(data);
    setStatus("Backtest ready. It only uses information available at each point in time.");
  };

  const runActive = (source) => (tab === "backtest" ? runBacktest(source) : runRegime(source)).catch((error) => setStatus(`Error: ${error.message}`));

  const runPortfolio = async (source) => {
    try {
      setStatus("Reading your portfolio...");
      const data = await api("/api/portfolio/diagnose", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ holdings: parsedHoldings(), source }) });
      setPortfolio(data);
      setStatus(`${source === "synthetic" ? "Sample" : "Live"} portfolio checkup is ready.`);
    } catch (error) { setStatus(`Portfolio error: ${error.message}`); }
  };

  const runResilience = async () => {
    try {
      setStatus("Running a 20-day stress check across 2,000 possible paths...");
      const data = await api("/api/portfolio/resilience", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ holdings: parsedHoldings(), source: "synthetic", simulations: 2000, horizon_days: 20 }) });
      setResilience(data);
      setStatus("Stress check ready.");
    } catch (error) { setStatus(`Resilience lab error: ${error.message}`); }
  };

  const ask = async (team = false) => {
    try {
      const endpoint = team ? "/api/research-team" : "/api/ask";
      const query = question.trim() || "What is happening in the market right now?";
      setResearch({ answer: team ? "Putting together a research brief..." : "Looking into that...", sources: [], note: "" });
      const data = await api(endpoint, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ query, symbol: symbol || "SPY", source: "synthetic" }) });
      setResearch({
        answer: team ? data.report : data.answer,
        sources: team ? (data.research_sources || []) : data.sources,
        note: team ? "This brief combines macro, risk, relationship, and research views." : (data.used_llm ? "Claude wrote this using the sources shown here." : "This answer was created from the local research library. Add ANTHROPIC_API_KEY to use Claude."),
      });
    } catch (error) { setResearch({ answer: `Error: ${error.message}`, sources: [], note: "" }); }
  };

  useEffect(() => { runRegime("synthetic").catch((error) => setStatus(`Error: ${error.message}`)); }, []);
  useEffect(() => {
    const protocol = location.protocol === "https:" ? "wss" : "ws";
    const socket = new WebSocket(`${protocol}://${location.host}/ws/regime?source=synthetic&speed=0.4&limit=400`);
    socket.onopen = () => setLive("Live updates on");
    socket.onmessage = ({ data }) => {
      const message = JSON.parse(data);
      if (message.type === "tick") setLive(`Live: ${message.label}, ${Math.round(message.confidence * 100)}% confidence, ${message.date}`);
      if (message.type === "complete") setLive("Live update finished");
    };
    socket.onerror = () => setLive("Live updates unavailable");
    return () => socket.close();
  }, []);

  const regimePlot = useMemo(() => {
    if (!regime) return null;
    const byLabel = {};
    regime.series.forEach((point) => {
      byLabel[point.label] ||= { x: [], y: [], text: [] };
      byLabel[point.label].x.push(point.date); byLabel[point.label].y.push(point.close); byLabel[point.label].text.push(`${point.label} - ${Math.round(point.confidence * 100)}%`);
    });
    return [{ x: regime.series.map((point) => point.date), y: regime.series.map((point) => point.close), type: "scatter", mode: "lines", line: { color: "rgba(139,151,168,.35)", width: 1 }, hoverinfo: "skip", showlegend: false }, ...Object.entries(byLabel).map(([label, points]) => ({ ...points, name: label, type: "scatter", mode: "markers", marker: { size: 5, color: COLORS[label] || "#888" }, hovertemplate: "%{x}<br>%{y:.2f}<br>%{text}<extra></extra>" }))];
  }, [regime]);

  const plotLayout = (title) => ({ paper_bgcolor: "transparent", plot_bgcolor: "transparent", font: { color: "#e6edf6" }, margin: { t: 10, r: 10, b: 40, l: 55 }, legend: { orientation: "h", y: 1.08 }, xaxis: { gridcolor: "#1e2838" }, yaxis: { gridcolor: "#1e2838", title } });

  return <>
    <header><div><span className="tag block pb-2 text-signal">Your portfolio, in context</span><h1>QuantMindAI</h1></div><span className="tag">Clearer market research<br />for everyday decisions</span><span className="tag live">{live}</span></header>
    <nav className="tabs">{[["regime", "Market regimes"], ["backtest", "Backtest"], ["portfolio", "Portfolio checkup"], ["research", "Research notes"]].map(([id, label]) => <button key={id} className={`tab ${tab === id ? "active" : ""}`} onClick={() => setTab(id)}>{label}</button>)}</nav>
    {!["portfolio", "research"].includes(tab) && <div className="controls">
      <input value={symbol} onChange={(event) => setSymbol(event.target.value)} placeholder="Symbol" size="8" />
      <select value={period} onChange={(event) => setPeriod(event.target.value)}><option value="1y">1y</option><option value="2y">2y</option><option value="5y">5y</option><option value="max">max</option></select>
      <select value={states} onChange={(event) => setStates(Number(event.target.value))}><option value="2">2 regimes</option><option value="3">3 regimes</option><option value="4">4 regimes</option></select>
      {tab === "backtest" && <label className="meta">cost <input value={cost} onChange={(event) => setCost(event.target.value)} size="3" /> bps</label>}
      <button className="primary" onClick={() => runActive("auto")}>Use live prices</button><button onClick={() => runActive("synthetic")}>Try the sample</button>
    </div>}
    <div className="status">{status}</div>
    {tab === "regime" && <main className="layout"><div><Panel title="Market right now">{regime ? <><div className={`label ${regime.current.label.toLowerCase()}`}>{regime.current.label}</div><div className="meta">{(regime.current.confidence * 100).toFixed(1)}% confidence on {regime.current.date}</div></> : <div className="meta">Getting the latest view...</div>}</Panel><Panel title="What the data shows"><table><tbody>{regime && Object.entries(regime.regime_stats).map(([label, value]) => <tr key={label}><td><span className={`badge ${label}`}>{label}</span></td><td className="num">{(value.share * 100).toFixed(0)}%</td><td className="num">{value.avg_return >= 0 ? "+" : ""}{value.avg_return}%</td><td className="num">{value.avg_vol}% vol</td></tr>)}</tbody></table></Panel></div><Panel title="Price history"><Chart data={regimePlot || []} layout={plotLayout("Price")} config={{ responsive: true, displayModeBar: false }} className="chart" /></Panel></main>}
    {tab === "backtest" && <main className="layout"><Panel title="Strategy check">{backtest ? <><div className="meta">This backtest made {backtest.audit.rebalances} decisions and used 0 future observations.</div><table><thead><tr><th>Metric</th><th className="num">Strategy</th><th className="num">Buy and hold</th></tr></thead><tbody>{backtestMetrics.map((metric) => { const format = (value) => pctMetrics.has(metric) ? `${(value * 100).toFixed(1)}%` : value.toFixed(2); return <tr key={metric}><td>{metric.replaceAll("_", " ")}</td><td className={`num ${backtest.strategy[metric] >= 0 ? "positive" : "negative"}`}>{format(backtest.strategy[metric])}</td><td className={`num ${backtest.benchmark[metric] >= 0 ? "positive" : "negative"}`}>{format(backtest.benchmark[metric])}</td></tr>; })}</tbody></table></> : <div className="meta">Run the backtest to compare the strategy with simply holding the asset.</div>}</Panel><Panel title="What one dollar would have become">{backtest && <Chart data={[{ x: backtest.equity_curve.map((point) => point.date), y: backtest.equity_curve.map((point) => point.strategy), name: "Strategy", type: "scatter", mode: "lines", line: { color: "#B7F36B", width: 2 } }, { x: backtest.equity_curve.map((point) => point.date), y: backtest.equity_curve.map((point) => point.benchmark), name: "Buy and hold", type: "scatter", mode: "lines", line: { color: "#8b97a8", width: 1.5, dash: "dot" } }]} layout={plotLayout("Growth of $1")} config={{ responsive: true, displayModeBar: false }} className="chart" />}</Panel></main>}
    {tab === "portfolio" && <main className="layout portfolio-layout"><Panel title="What do you own?"><label className="meta">Type holdings like SPY:60, TLT:40</label><input className="holdings" value={holdings} onChange={(event) => setHoldings(event.target.value)} /><div className="buttons"><button className="primary" onClick={() => runPortfolio("auto")}>Use live prices</button><button onClick={() => runPortfolio("synthetic")}>Try the sample</button><button onClick={runResilience}>Run a stress check</button></div><p className="meta">This is a historical checkup, not personal investment advice.</p></Panel><Panel title="Your portfolio at a glance">{portfolio ? <><div className="label">{portfolio.summary.diversification_score}/100</div><div className="meta">A diversification score based on {portfolio.observations} days of history</div><table><tbody>{[["Largest holding", `${(portfolio.summary.top_holding_weight * 100).toFixed(0)}%`], ["Effective number of holdings", portfolio.summary.effective_holdings], ["Average correlation", portfolio.summary.average_pairwise_correlation], ["Worst historical drawdown", `${(portfolio.summary.max_drawdown * 100).toFixed(1)}%`], ["Typical bad day (95% VaR)", `${(portfolio.summary.historical_var_95_daily * 100).toFixed(2)}%`]].map(([label, value]) => <tr key={label}><td>{label}</td><td className="num">{value}</td></tr>)}</tbody></table>{portfolio.flags.map((flag) => <div className={`flag ${flag.level}`} key={flag.code}><b>{flag.code.replaceAll("_", " ")}</b><br /><span className="meta">{flag.message}</span></div>)}</> : <div className="meta">Run a quick checkup to see where the risk is coming from.</div>}</Panel><Panel title="Stress check" className="wide">{resilience ? <Resilience data={resilience} /> : <div className="meta">See which holdings drive risk, how rough a 20-day stretch could be, and how equal weighting compares.</div>}</Panel></main>}
    {tab === "research" && <main className="layout"><Panel title="Sources used">{research.sources.length ? research.sources.map((source) => <div className="meta" key={source.id || source.title}>- <b>{source.title}</b> [{source.id}]</div>) : <div className="meta">Ask a question and the sources behind the answer will show up here.</div>}</Panel><Panel title="Research notes"><div className="question"><input value={question} onChange={(event) => setQuestion(event.target.value)} onKeyDown={(event) => event.key === "Enter" && ask()} placeholder="What changed in this market regime?" /><button className="primary" onClick={() => ask()}>Ask a question</button><button onClick={() => ask(true)}>Build a research brief</button></div><div className="answer">{research.answer}</div><p className="meta">{research.note}</p></Panel></main>}
  </>;
}

function Resilience({ data }) {
  const current = data.current_allocation;
  const equal = data.equal_weight_counterfactual;
  const volatilityChange = data.comparison.annualized_volatility_change;
  return <><div className="label">{(current.p05_horizon_return * 100).toFixed(1)}%</div><div className="meta">A tough-but-plausible {data.simulation.horizon_days}-day outcome from {data.simulation.simulations.toLocaleString()} historical simulations</div><table><thead><tr><th>Holding</th><th className="num">Weight</th><th className="num">Share of risk</th></tr></thead><tbody>{current.risk_attribution.map((item) => <tr key={item.symbol}><td>{item.symbol}</td><td className="num">{(item.weight * 100).toFixed(1)}%</td><td className="num">{(item.risk_contribution * 100).toFixed(1)}%</td></tr>)}</tbody></table><table><tbody><tr><td>Chance of losing money over 20 days</td><td className="num">{(current.probability_of_loss * 100).toFixed(1)}%</td></tr><tr><td>Average loss in the worst 5% of outcomes</td><td className="num">{(current.cvar_95_horizon_return * 100).toFixed(1)}%</td></tr><tr><td>Change in volatility with equal weights</td><td className="num">{volatilityChange >= 0 ? "+" : ""}{(volatilityChange * 100).toFixed(1)} pts</td></tr><tr><td>Equal-weight downside outcome</td><td className="num">{(equal.p05_horizon_return * 100).toFixed(1)}%</td></tr></tbody></table><p className="meta">Based on the portfolio's own history. Equal weighting is just a comparison, not a recommendation.</p></>;
}

createRoot(document.getElementById("root")).render(<App />);
