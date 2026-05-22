import numpy as np
import torch

from quantmind.graph import (
    GAT,
    build_adjacency,
    generate_multi_asset_returns,
    graph_summary,
    node_features,
    train_sector_gat,
)


def test_synthetic_universe_and_graph():
    returns, sectors = generate_multi_asset_returns(n_assets=30, n_sectors=3, n_days=400, seed=0)
    assert returns.shape == (400, 30)
    assert set(np.unique(sectors)) == {0, 1, 2}

    adj, nodes = build_adjacency(returns, threshold=0.5)
    assert adj.shape == (30, 30)
    assert (np.diag(adj) == 1.0).all()  # self-loops present

    feats = node_features(returns)
    assert feats.shape == (30, 3)

    summary = graph_summary(returns, threshold=0.5)
    assert summary["n_edges"] > 0
    assert summary["n_nodes"] == 30


def test_gat_forward_shape():
    net = GAT(in_dim=3, hidden=8, n_classes=3)
    x = torch.randn(10, 3)
    adj = torch.eye(10)
    out = net(x, adj)
    assert out.shape == (10, 3)


def test_gat_recovers_sectors_from_graph():
    returns, sectors = generate_multi_asset_returns(n_assets=30, n_sectors=3, n_days=500, seed=0)
    _, metrics = train_sector_gat(returns, sectors, epochs=300, seed=0)

    # 3 sectors => random guessing is ~0.33; the graph should push well past that.
    assert metrics["test_accuracy"] > 0.6, metrics
