"""GCN / SAGE / GAT baselines on the 64x64 grid.

Each model has a plain variant and an edge-aware (+e) variant that uses
wind/slope/NDVI edge features. Input (B, 12, 64, 64), output (B, 64, 64).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, MessagePassing, SAGEConv

from ..graph import N as N_NODES
from ..graph import (
    batched_edge_index,
    compute_edge_features,
    grid_to_nodes,
)


# --- edge-aware conv layers ---

class EdgeWeightedGCNConv(nn.Module):
    """GCN with a per-edge scalar weight derived from edge features.

    Cheapest way to bolt edge info onto GCN: tiny MLP -> sigmoid -> edge_weight.
    """

    def __init__(self, hid: int, edge_dim: int = 3):
        super().__init__()
        self.conv = GCNConv(hid, hid, add_self_loops=True, normalize=True)
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_dim, hid // 2), nn.SiLU(),
            nn.Linear(hid // 2, 1),
        )

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor) -> torch.Tensor:
        w = torch.sigmoid(self.edge_mlp(edge_attr)).squeeze(-1)  # (E,)
        return self.conv(h, edge_index, edge_weight=w)


class EdgeFeatSAGEConv(MessagePassing):
    """SAGE with edge features concatenated into the message.

    msg(j -> i) = W_msg * [h_j || edge_attr_ij]
    """

    def __init__(self, hid: int, edge_dim: int = 3):
        super().__init__(aggr="mean")
        self.lin_self = nn.Linear(hid, hid)
        self.lin_msg = nn.Linear(hid + edge_dim, hid)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor) -> torch.Tensor:
        return self.lin_self(h) + self.propagate(edge_index, x=h, edge_attr=edge_attr)

    def message(self, x_j: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        return self.lin_msg(torch.cat([x_j, edge_attr], dim=-1))


# --- stack wrappers ---

class _GridGNN(nn.Module):
    """encode -> n_layers conv -> decode. No edge features."""

    def __init__(self, conv_cls, edge_index, in_dim: int = 12, hid: int = 64,
                 n_layers: int = 3, dropout: float = 0.1, **conv_kw):
        super().__init__()
        self.register_buffer("edge_index", edge_index, persistent=False)
        self.in_proj = nn.Linear(in_dim, hid)
        self.convs = nn.ModuleList(
            [conv_cls(hid, hid, **conv_kw) for _ in range(n_layers)]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(hid) for _ in range(n_layers)])
        self.head = nn.Linear(hid, 1)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        nodes = grid_to_nodes(x)
        h = self.in_proj(nodes).reshape(B * N_NODES, -1)
        ei = batched_edge_index(self.edge_index, B, N_NODES)
        for conv, norm in zip(self.convs, self.norms):
            h_new = F.dropout(F.relu(norm(conv(h, ei))),
                              p=self.dropout, training=self.training)
            h = h + h_new   # residual
        return self.head(h).view(B, 64, 64)


class _EdgeAwareGridGNN(nn.Module):
    """Same as _GridGNN but edges carry features. uniform_edges=True zeros them."""

    def __init__(
        self,
        layer_cls,
        edge_index: torch.Tensor,
        edge_dirs: torch.Tensor,
        in_dim: int = 12,
        hid: int = 64,
        edge_dim: int = 3,
        n_layers: int = 3,
        dropout: float = 0.1,
        uniform_edges: bool = False,
        norm_mean: torch.Tensor | None = None,
        norm_std: torch.Tensor | None = None,
        **layer_kw,
    ):
        super().__init__()
        self.register_buffer("edge_index", edge_index, persistent=False)
        self.register_buffer("edge_dirs", edge_dirs, persistent=False)
        if norm_mean is not None:
            self.register_buffer("norm_mean", torch.as_tensor(norm_mean, dtype=torch.float32),
                                 persistent=False)
            self.register_buffer("norm_std", torch.as_tensor(norm_std, dtype=torch.float32),
                                 persistent=False)
        else:
            self.norm_mean = None
            self.norm_std = None
        self.in_proj = nn.Linear(in_dim, hid)
        self.layers = nn.ModuleList(
            [layer_cls(hid, edge_dim=edge_dim, **layer_kw) for _ in range(n_layers)]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(hid) for _ in range(n_layers)])
        self.head = nn.Linear(hid, 1)
        self.dropout = dropout
        self.uniform_edges = uniform_edges

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        E = self.edge_index.size(1)
        nodes = grid_to_nodes(x)
        h = self.in_proj(nodes).reshape(B * N_NODES, -1)
        edge_attr = compute_edge_features(
            nodes, self.edge_index, self.edge_dirs,
            norm_mean=self.norm_mean, norm_std=self.norm_std,
            uniform=self.uniform_edges,
        ).reshape(B * E, -1)
        ei = batched_edge_index(self.edge_index, B, N_NODES)
        for layer, lnorm in zip(self.layers, self.norms):
            h_new = F.dropout(F.relu(lnorm(layer(h, ei, edge_attr))),
                              p=self.dropout, training=self.training)
            h = h + h_new
        return self.head(h).view(B, 64, 64)


# --- plain (no edge features) ---

class GridGCN(_GridGNN):
    def __init__(self, edge_index, **kw):
        super().__init__(GCNConv, edge_index, add_self_loops=True, **kw)


class GridSAGE(_GridGNN):
    def __init__(self, edge_index, **kw):
        super().__init__(SAGEConv, edge_index, **kw)


class GridGAT(_GridGNN):
    def __init__(self, edge_index, heads: int = 4, **kw):
        # concat=False so out dim stays at hid (4-head average)
        super().__init__(GATConv, edge_index, heads=heads, concat=False, dropout=0.1, **kw)


# --- edge-aware (+e) variants ---

class GridGCNEdge(_EdgeAwareGridGNN):
    def __init__(self, edge_index, edge_dirs, **kw):
        super().__init__(EdgeWeightedGCNConv, edge_index, edge_dirs, **kw)


class GridSAGEEdge(_EdgeAwareGridGNN):
    def __init__(self, edge_index, edge_dirs, **kw):
        super().__init__(EdgeFeatSAGEConv, edge_index, edge_dirs, **kw)


class _GATEdgeLayer(nn.Module):
    # tiny adapter so GATConv fits the (h, ei, edge_attr) signature
    def __init__(self, hid: int, edge_dim: int = 3, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.conv = GATConv(hid, hid, heads=heads, concat=False,
                            edge_dim=edge_dim, dropout=dropout, add_self_loops=False)

    def forward(self, h, edge_index, edge_attr):
        return self.conv(h, edge_index, edge_attr=edge_attr)


class GridGATEdge(_EdgeAwareGridGNN):
    def __init__(self, edge_index, edge_dirs, heads: int = 4, **kw):
        super().__init__(_GATEdgeLayer, edge_index, edge_dirs, heads=heads, **kw)
