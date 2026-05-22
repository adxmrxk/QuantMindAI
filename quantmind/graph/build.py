"""Build an asset-correlation graph and its node features.

Includes a synthetic multi-asset generator with explicit market + sector factor
structure, so the correlation graph has a known sector ground truth to learn and
the whole module runs offline.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd


def generate_multi_asset_returns(
    n_assets: int = 30,
    n_sectors: int = 3,
    n_days: int = 500,
    seed: int = 0,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Synthetic daily returns with market + sector factor structure.

    Each asset = market_beta·market + sector_beta·sector_factor + idiosyncratic.
    Returns (returns_df [n_days × n_assets], sector_label_per_asset).
    """
    rng = np.random.default_rng(seed)
    sectors = np.array([i % n_sectors for i in range(n_assets)])
    sector_drift = np.linspace(-0.0006, 0.0009, n_sectors)

    market = rng.normal(0.0, 0.010, n_days)
    sector_factors = {k: rng.normal(sector_drift[k], 0.013, n_days) for k in range(n_sectors)}

    cols = {}
    for i in range(n_assets):
        k = int(sectors[i])
        beta_m = rng.uniform(0.3, 0.7)
        beta_s = rng.uniform(0.9, 1.2)
        idio = rng.normal(0.0, 0.006, n_days)
        cols[f"A{i:02d}"] = beta_m * market + beta_s * sector_factors[k] + idio

    return pd.DataFrame(cols), sectors


def correlation_matrix(returns_df: pd.DataFrame) -> np.ndarray:
    return returns_df.corr().to_numpy()


def build_adjacency(
    returns_df: pd.DataFrame,
    threshold: float = 0.5,
    self_loops: bool = True,
) -> tuple[np.ndarray, list[str]]:
    """Adjacency where |corr| >= threshold. Self-loops keep every node well-defined."""
    corr = correlation_matrix(returns_df)
    adj = (np.abs(corr) >= threshold).astype("float32")
    np.fill_diagonal(adj, 1.0 if self_loops else 0.0)
    return adj, list(returns_df.columns)


def node_features(returns_df: pd.DataFrame) -> np.ndarray:
    """Per-asset summary stats. Individually weak for sector ID — the graph carries
    most of the structure — which is the point of using a GNN."""
    r = returns_df
    feats = np.stack(
        [
            r.mean().to_numpy() * 100.0,
            r.std().to_numpy() * 100.0,
            r.skew().to_numpy(),
        ],
        axis=1,
    )
    return feats.astype("float32")


def graph_summary(returns_df: pd.DataFrame, threshold: float = 0.5) -> dict:
    """Edge/degree/component stats via networkx (no self-loops)."""
    adj, nodes = build_adjacency(returns_df, threshold, self_loops=False)
    g = nx.from_numpy_array(adj)
    degrees = [d for _, d in g.degree()]
    return {
        "n_nodes": g.number_of_nodes(),
        "n_edges": g.number_of_edges(),
        "avg_degree": round(float(np.mean(degrees)) if degrees else 0.0, 2),
        "n_components": nx.number_connected_components(g),
    }
