"""Standard GNN baselines (GCN / GraphSAGE / GAT) on the 64x64 grid graph.

All wrappers take input x (B, 12, 64, 64) and return logits (B, 64, 64).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, SAGEConv

from ..graph import N as N_NODES
from ..graph import batched_edge_index, grid_to_nodes


class _GridGNN(nn.Module):
    """Generic wrapper: encode -> n_layers of conv -> decode to per-node logit.

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
