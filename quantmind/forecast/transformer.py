"""A compact Transformer encoder for sequence classification.

Deliberately small (a couple of layers, d_model=32) so it trains in seconds on
CPU — the point is a correct, end-to-end time-series Transformer, not a
parameter-count contest. Swapping in PatchTST / Temporal Fusion Transformer is a
drop-in replacement behind the same ``forward`` contract.
"""

from __future__ import annotations

import torch
from torch import nn


class RegimeTransformer(nn.Module):
    def __init__(
        self,
        n_features: int = 3,
        seq_len: int = 30,
        d_model: int = 32,
        nhead: int = 4,
        num_layers: int = 2,
        n_classes: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.input_proj = nn.Linear(n_features, d_model)
        # Learned positional encoding (one vector per timestep).
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len, d_model))
        nn.init.normal_(self.pos_embedding, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, n_features) -> logits (batch, n_classes)."""
        h = self.input_proj(x) + self.pos_embedding[:, : x.size(1)]
        h = self.encoder(h)
        return self.head(h[:, -1])  # classify from the final timestep
