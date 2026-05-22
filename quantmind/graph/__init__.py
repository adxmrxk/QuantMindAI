"""Graph neural networks for asset relationship mapping.

Assets are nodes; edges connect assets whose returns are correlated. A Graph
Attention Network then learns over that structure — here, recovering each
asset's sector purely from weakly-informative node features plus the correlation
graph, which demonstrates the model exploiting *relationships*, not just
per-asset signals. The same machinery extends to contagion / spillover
prediction.
"""

from quantmind.graph.build import (
    build_adjacency,
    correlation_matrix,
    generate_multi_asset_returns,
    graph_summary,
    node_features,
)
from quantmind.graph.gat import GAT, GraphAttentionLayer
from quantmind.graph.train import train_sector_gat

__all__ = [
    "generate_multi_asset_returns",
    "correlation_matrix",
    "build_adjacency",
    "node_features",
    "graph_summary",
    "GAT",
    "GraphAttentionLayer",
    "train_sector_gat",
]
