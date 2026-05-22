"""Transductive node-classification training for the sector-recovery task."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn

from quantmind.graph.build import build_adjacency, node_features
from quantmind.graph.gat import GAT


def train_sector_gat(
    returns_df: pd.DataFrame,
    sectors: np.ndarray,
    threshold: float = 0.5,
    hidden: int = 16,
    epochs: int = 300,
    lr: float = 0.01,
    train_frac: float = 0.6,
    seed: int = 0,
) -> tuple[GAT, dict]:
    """Train a GAT to recover asset sectors from features + the correlation graph.

    Transductive: the whole graph is visible, but the loss is computed only on a
    training subset of nodes; accuracy is reported on the held-out nodes.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    adj, nodes = build_adjacency(returns_df, threshold=threshold)
    feats = node_features(returns_df)
    feats = (feats - feats.mean(axis=0)) / (feats.std(axis=0) + 1e-8)

    x = torch.tensor(feats, dtype=torch.float32)
    a = torch.tensor(adj, dtype=torch.float32)
    y = torch.tensor(sectors, dtype=torch.long)

    n = len(nodes)
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    n_train = int(n * train_frac)
    train_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[order[:n_train]] = True
    test_mask = ~train_mask

    n_classes = int(sectors.max()) + 1
    model = GAT(in_dim=x.shape[1], hidden=hidden, n_classes=n_classes)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    loss_fn = nn.CrossEntropyLoss()

    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        out = model(x, a)
        loss_fn(out[train_mask], y[train_mask]).backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        pred = model(x, a).argmax(dim=1)
        test_acc = float((pred[test_mask] == y[test_mask]).float().mean())

    metrics = {
        "test_accuracy": round(test_acc, 4),
        "n_nodes": n,
        "n_classes": n_classes,
        "n_train": int(n_train),
        "n_test": int(test_mask.sum()),
    }
    return model, metrics
