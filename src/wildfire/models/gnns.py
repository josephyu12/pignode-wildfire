"""Standard GNN baselines (GCN / GraphSAGE / GAT) on the 64x64 grid graph.

Each model exposes two variants -- a plain version that ignores edge attributes
and an *edge-aware* version that consumes the proposal's wind/slope/NDVI edge
features (proposal §4.1, §5 challenge #3). For GAT the edge-aware path uses
PyG's native `edge_dim` argument (eq. 3 in the proposal). For GCN and SAGE we
manually inject edge features into message passing because those layers lack
native edge-attribute support; see `EdgeWeightedGCNConv` and `EdgeFeatSAGEConv`.

All wrappers take input x (B, 12, 64, 64) and return logits (B, 64, 64).
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


# -------------------------- Edge-aware conv layers --------------------------

class EdgeWeightedGCNConv(nn.Module):
    """GCN with a scalar per-edge weight derived from physics edge features.

    GCNConv accepts an `edge_weight` argument (used directly in symmetric
    normalization). We map the 3-d edge feature vector through a tiny MLP and
    a sigmoid to produce a positive scalar in (0, 1) per edge, which scales
    that edge's contribution. This is the lightest-touch way to make GCN
    edge-aware without rewriting the message passing recipe.
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
    """GraphSAGE-style aggregation with edge features concatenated into messages.

    Standard SAGE: h_i' = W_self * h_i + W_neigh * mean_{j in N(i)} h_j.
    Edge-aware variant: messages from j to i are W_msg * [h_j || edge_attr_ij],
    so wind/slope/NDVI directly modulate what each neighbor contributes.
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


# -------------------------- Generic stack wrappers --------------------------

class _GridGNN(nn.Module):
    """Plain (no-edge-features) wrapper: encode -> n_layers conv -> decode.

    edge_index is registered as a buffer so it moves with .to(device).
    """

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
        nodes = grid_to_nodes(x)                      # (B, N, F_in)
        h = self.in_proj(nodes).reshape(B * N_NODES, -1)  # (B*N, hid)
        ei = batched_edge_index(self.edge_index, B, N_NODES)
        for conv, norm in zip(self.convs, self.norms):
            h_new = conv(h, ei)
            h_new = F.relu(norm(h_new))
            h_new = F.dropout(h_new, p=self.dropout, training=self.training)
            h = h + h_new   # residual stabilizes deeper stacks
        logit = self.head(h).view(B, 64, 64)
        return logit


class _EdgeAwareGridGNN(nn.Module):
    """Wrapper for layers that accept (h, edge_index, edge_attr).

    Computes per-batch edge features once per forward pass (matching the
    PI-GNODE pipeline so the §4.2 "do edges matter for non-attention GNNs?"
    comparison is honest). Setting `uniform_edges=True` zeroes the edge
    feature tensor for the edge-encoding ablation.
    """

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
        nodes = grid_to_nodes(x)                                 # (B, N, F)
        h = self.in_proj(nodes).reshape(B * N_NODES, -1)         # (B*N, hid)
        edge_attr = compute_edge_features(
            nodes, self.edge_index, self.edge_dirs,
            norm_mean=self.norm_mean, norm_std=self.norm_std,
            uniform=self.uniform_edges,
        ).reshape(B * E, -1)
        ei = batched_edge_index(self.edge_index, B, N_NODES)
        for layer, lnorm in zip(self.layers, self.norms):
            h_new = layer(h, ei, edge_attr)
            h_new = F.relu(lnorm(h_new))
            h_new = F.dropout(h_new, p=self.dropout, training=self.training)
            h = h + h_new
        return self.head(h).view(B, 64, 64)


# -------------------------- Plain (no-edge-feature) baselines --------------

class GridGCN(_GridGNN):
    def __init__(self, edge_index, **kw):
        super().__init__(GCNConv, edge_index, add_self_loops=True, **kw)


class GridSAGE(_GridGNN):
    def __init__(self, edge_index, **kw):
        super().__init__(SAGEConv, edge_index, **kw)


class GridGAT(_GridGNN):
    def __init__(self, edge_index, heads: int = 4, **kw):
        # GATConv outputs heads*out_dim concatenated by default; force concat=False
        # so output dim stays = hid. Use 4 heads for averaging.
        super().__init__(
            GATConv, edge_index, heads=heads, concat=False, dropout=0.1, **kw
        )


# -------------------------- Edge-aware (proposal §4.2 +e) ------------------

class GridGCNEdge(_EdgeAwareGridGNN):
    """GCN with learned edge weights from physics edge features."""

    def __init__(self, edge_index, edge_dirs, **kw):
        super().__init__(EdgeWeightedGCNConv, edge_index, edge_dirs, **kw)


class GridSAGEEdge(_EdgeAwareGridGNN):
    """GraphSAGE with edge features concatenated into messages."""

    def __init__(self, edge_index, edge_dirs, **kw):
        super().__init__(EdgeFeatSAGEConv, edge_index, edge_dirs, **kw)


class _GATEdgeLayer(nn.Module):
    """Adapter so GATConv plugs into the EdgeAwareGridGNN ctor signature."""

    def __init__(self, hid: int, edge_dim: int = 3, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.conv = GATConv(hid, hid, heads=heads, concat=False,
                            edge_dim=edge_dim, dropout=dropout, add_self_loops=False)

    def forward(self, h, edge_index, edge_attr):
        return self.conv(h, edge_index, edge_attr=edge_attr)


class GridGATEdge(_EdgeAwareGridGNN):
    """GAT with native edge_dim attention (proposal eq. 3)."""

    def __init__(self, edge_index, edge_dirs, heads: int = 4, **kw):
        super().__init__(_GATEdgeLayer, edge_index, edge_dirs, heads=heads, **kw)
