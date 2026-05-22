"""A Graph Attention Network implemented directly in PyTorch.

Written from scratch (rather than via torch-geometric) to keep the dependency
footprint small and the attention mechanism legible. Operates full-batch on a
dense adjacency — fine for the asset-universe sizes here.
"""

from __future__ import annotations

import torch
from torch import nn


class GraphAttentionLayer(nn.Module):
    """Single-head graph attention (Velickovic et al., 2018)."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1, alpha: float = 0.2) -> None:
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a_src = nn.Parameter(torch.empty(out_dim))
        self.a_dst = nn.Parameter(torch.empty(out_dim))
        self.leaky_relu = nn.LeakyReLU(alpha)
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.zeros_(self.a_src)
        nn.init.zeros_(self.a_dst)

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # h: (N, in_dim), adj: (N, N) with 0/1 entries (incl. self-loops).
        wh = self.W(h)  # (N, out_dim)
        # Additive attention scores e_ij = LeakyReLU(a_src·Wh_i + a_dst·Wh_j).
        scores = (wh @ self.a_src).unsqueeze(1) + (wh @ self.a_dst).unsqueeze(0)
        scores = self.leaky_relu(scores)
        scores = scores.masked_fill(adj <= 0, float("-inf"))
        attn = torch.softmax(scores, dim=1)
        attn = self.dropout(attn)
        return attn @ wh  # (N, out_dim)


class GAT(nn.Module):
    def __init__(self, in_dim: int, hidden: int, n_classes: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.gat1 = GraphAttentionLayer(in_dim, hidden, dropout)
        self.gat2 = GraphAttentionLayer(hidden, n_classes, dropout)
        self.elu = nn.ELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = self.dropout(self.elu(self.gat1(x, adj)))
        return self.gat2(h, adj)  # logits (N, n_classes)
