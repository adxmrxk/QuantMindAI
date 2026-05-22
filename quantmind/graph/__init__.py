"""Graph neural networks for asset relationship mapping.

Assets are nodes; edges connect assets whose returns are correlated. A Graph
Attention Network then learns over that structure — here, recovering each
asset's sector purely from weakly-informative node features plus the correlation
graph, which demonstrates the model exploiting *relationships*, not just
per-asset signals. The same machinery extends to contagion / spillover
prediction.

The graph-building helpers (numpy / networkx) import eagerly; the PyTorch GAT
and its trainer are loaded lazily on first access, so importing this package —
e.g. from the API for ``graph_summary`` — does not pull in PyTorch.
"""

import importlib

from quantmind.graph.build import (
    build_adjacency,
    correlation_matrix,
    generate_multi_asset_returns,
    graph_summary,
    node_features,
)

# name -> module providing it (imported on demand to keep torch optional here)
_LAZY = {
    "GAT": "quantmind.graph.gat",
    "GraphAttentionLayer": "quantmind.graph.gat",
    "train_sector_gat": "quantmind.graph.train",
}


def __getattr__(name):  # PEP 562 module-level lazy attributes
    if name in _LAZY:
        return getattr(importlib.import_module(_LAZY[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
