"""Physics-Informed Graph Neural ODE (PI-GNODE) for wildfire spread.

Architecture
------------
1. Encode 12 input features per node -> hidden state h(0).
2. ODE: dh/dt = f_θ(h, A, e) where f_θ is a GAT layer with edge-conditioned
   attention (edge features = wind-edge alignment, slope, NDVI continuity).
3. Integrate h(T) using torchdiffeq adjoint (default: dopri5).
4. Decode h(T) -> per-node fire logit.
5. Monotonicity: predicted prob is forced >= initial fire prob (burn irreversibility).

Following equations (1)-(4) in the project proposal.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
from torchdiffeq import odeint, odeint_adjoint

from ..graph import (
    N as N_NODES,
    batched_edge_index,
    compute_edge_features,
    grid_to_nodes,
)


class GATODEFunc(nn.Module):
    """Time-derivative dh/dt parameterized by edge-conditioned GAT.

    PyG's GATConv with edge_dim>0 implements:
        α_ij = softmax(LeakyReLU(a^T [Wh_i || Wh_j || W_e e_ij]))
    which matches eq. (3) in the proposal exactly.
    """

    def __init__(self, hidden: int = 64, edge_dim: int = 3, heads: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        self.gat = GATConv(
            hidden, hidden, heads=heads, concat=False,
            edge_dim=edge_dim, dropout=dropout, add_self_loops=False,
        )
        self.norm = nn.LayerNorm(hidden)
        self.act = nn.SiLU()
        self._edge_index: torch.Tensor | None = None
        self._edge_attr: torch.Tensor | None = None
        self.nfe: int = 0   # number-of-function-evaluations counter (reported in paper)

    def set_context(self, edge_index: torch.Tensor, edge_attr: torch.Tensor):
        self._edge_index = edge_index
        self._edge_attr = edge_attr
        self.nfe = 0

    def forward(self, t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        self.nfe += 1
        msg = self.gat(h, self._edge_index, edge_attr=self._edge_attr)
        return self.act(self.norm(msg))


class PIGNODE(nn.Module):
    def __init__(
        self,
        edge_index: torch.Tensor,
        edge_dirs: torch.Tensor,
        in_dim: int = 12,
        hidden: int = 64,
        edge_dim: int = 3,
        heads: int = 4,
        t_end: float = 1.0,
        n_eval_steps: int = 2,
        monotone: bool = True,
        uniform_edges: bool = False,
        norm_mean: torch.Tensor | None = None,
        norm_std: torch.Tensor | None = None,
        solver: str = "dopri5",
        adjoint: bool = True,
        rtol: float = 1e-3,
        atol: float = 1e-3,
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

        self.encode = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )
        self.f = GATODEFunc(hidden, edge_dim=edge_dim, heads=heads)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        self.register_buffer(
            "t", torch.linspace(0.0, t_end, n_eval_steps + 1), persistent=False
        )
        self.monotone = monotone
        self.uniform_edges = uniform_edges
        self.solver = solver
        self.adjoint = adjoint
        self.rtol = rtol
        self.atol = atol

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        E = self.edge_index.size(1)
        nodes = grid_to_nodes(x)                                 # (B, N, F)
        h0 = self.encode(nodes).reshape(B * N_NODES, -1)         # (B*N, hid)

        edge_attr = compute_edge_features(
            nodes, self.edge_index, self.edge_dirs,
            norm_mean=self.norm_mean, norm_std=self.norm_std,
            uniform=self.uniform_edges,
        ).reshape(B * E, 3)
        ei = batched_edge_index(self.edge_index, B, N_NODES)
        self.f.set_context(ei, edge_attr)

        if self.adjoint:
            h_traj = odeint_adjoint(
                self.f, h0, self.t, method=self.solver,
                rtol=self.rtol, atol=self.atol,
                adjoint_params=tuple(self.f.parameters()),
            )
        else:
            h_traj = odeint(
                self.f, h0, self.t, method=self.solver,
                rtol=self.rtol, atol=self.atol,
            )
        h_T = h_traj[-1]
        logits = self.head(h_T).view(B, 64, 64)

        if self.monotone:
            # Eq. (4): predicted prob >= initial fire prob (burn irreversibility).
            # Implemented as: where init==1, force logit to a large positive value.
            init_fire = x[:, 0]   # PrevFireMask, normalized but binary-valued.
            big = torch.full_like(logits, 6.0)   # σ(6) ≈ 0.998
            logits = torch.where(init_fire > 0.5, torch.maximum(logits, big), logits)

        return logits
