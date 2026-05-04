"""PI-GNODE: physics-informed graph neural ODE for wildfire spread.

encode -> integrate dh/dt = GAT_theta(h, edges) -> decode -> monotonicity floor.
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
    """dh/dt as a stack of edge-conditioned GAT layers.

    Each call does `ode_layers` hops of message passing, so a T-step solver
    gets T*ode_layers hops total. Set record_norm=True before the solve to
    capture per-step ||dh/dt||^2 for the Frobenius regularizer.
    """

    def __init__(self, hidden: int = 64, edge_dim: int = 3, heads: int = 4,
                 dropout: float = 0.1, ode_layers: int = 2):
        super().__init__()
        self.gats = nn.ModuleList([
            GATConv(hidden, hidden, heads=heads, concat=False,
                    edge_dim=edge_dim, dropout=dropout, add_self_loops=False)
            for _ in range(ode_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(ode_layers)])
        self.act = nn.SiLU()
        # identity-ish init: shrink the last GAT so dh/dt starts near zero
        with torch.no_grad():
            self.gats[-1].lin.weight.mul_(0.1)
        self._edge_index: torch.Tensor | None = None
        self._edge_attr: torch.Tensor | None = None
        self.nfe: int = 0
        self.record_norm: bool = False
        self.dyn_norms: list[torch.Tensor] = []

    def set_context(self, edge_index: torch.Tensor, edge_attr: torch.Tensor,
                    record_norm: bool = False):
        self._edge_index = edge_index
        self._edge_attr = edge_attr
        self.nfe = 0
        self.record_norm = record_norm
        self.dyn_norms = []

    def forward(self, t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        self.nfe += 1
        z = h
        for gat, norm in zip(self.gats, self.norms):
            z = self.act(norm(gat(z, self._edge_index, edge_attr=self._edge_attr)))
        if self.record_norm:
            # mean-squared per element; size-invariant proxy for ||dh/dt||_F^2
            self.dyn_norms.append(z.pow(2).mean())
        return z


class PIGNODE(nn.Module):
    def __init__(
        self,
        edge_index: torch.Tensor,
        edge_dirs: torch.Tensor,
        in_dim: int = 12,
        hidden: int = 64,
        edge_dim: int = 3,
        heads: int = 4,
        ode_layers: int = 2,
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
        return_dyn_norms: bool = False,
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
        self.f = GATODEFunc(hidden, edge_dim=edge_dim, heads=heads, ode_layers=ode_layers)
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
        self.return_dyn_norms = return_dyn_norms
        self.t_end = t_end
        self.n_eval_steps = n_eval_steps

    def _prepare_context(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # encode + batched edges + edge features. Shared by single-day and rollout.
        B = x.size(0)
        E = self.edge_index.size(1)
        nodes = grid_to_nodes(x)                              # (B, N, F)
        h0 = self.encode(nodes).reshape(B * N_NODES, -1)
        edge_attr = compute_edge_features(
            nodes, self.edge_index, self.edge_dirs,
            norm_mean=self.norm_mean, norm_std=self.norm_std,
            uniform=self.uniform_edges,
        ).reshape(B * E, 3)
        ei = batched_edge_index(self.edge_index, B, N_NODES)
        return h0, ei, edge_attr

    def _integrate(self, h0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if self.adjoint:
            return odeint_adjoint(
                self.f, h0, t, method=self.solver,
                rtol=self.rtol, atol=self.atol,
                adjoint_params=tuple(self.f.parameters()),
            )
        return odeint(self.f, h0, t, method=self.solver, rtol=self.rtol, atol=self.atol)

    def _apply_monotone(self, logits: torch.Tensor, init_fire: torch.Tensor) -> torch.Tensor:
        # Hard floor: cells already burning -> p ~ 1. Inference only;
        # at train time this would zero-out gradient on the densest label-1
        # cells, so we use the soft penalty in losses.py instead.
        big = torch.full_like(logits, 6.0)   # sigmoid(6) ~ 0.998
        return torch.where(init_fire > 0.5, torch.maximum(logits, big), logits)

    def forward(self, x: torch.Tensor):
        h0, ei, edge_attr = self._prepare_context(x)
        record = self.return_dyn_norms and self.training
        self.f.set_context(ei, edge_attr, record_norm=record)

        h_traj = self._integrate(h0, self.t)
        h_T = h_traj[-1]
        logits = self.head(h_T).view(x.size(0), 64, 64)

        if self.monotone and not self.training:
            logits = self._apply_monotone(logits, x[:, 0])

        if self.return_dyn_norms and self.training:
            return logits, list(self.f.dyn_norms)
        return logits

    def forward_rollout(
        self,
        x: torch.Tensor,
        n_days: int,
        teacher_fire_masks: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Multi-day forecast.

        Integrates one unit-day step at a time, decoding fire prob at each
        integer day. If teacher_fire_masks is given, we substitute the GT
        fire mask back into channel 0 at each boundary; otherwise we feed
        back our own thresholded prediction.

        Returns logits of shape (B, n_days, H, W).
        """
        assert n_days >= 1
        B = x.size(0)
        device = x.device
        outs: list[torch.Tensor] = []

        running_fire = x[:, 0].clone()
        cur_x = x

        for d in range(1, n_days + 1):
            h0, ei, edge_attr = self._prepare_context(cur_x)
            self.f.set_context(ei, edge_attr, record_norm=False)
            t = torch.tensor([0.0, 1.0], device=device, dtype=h0.dtype)
            h_T = self._integrate(h0, t)[-1]
            logits = self.head(h_T).view(B, 64, 64)

            if self.monotone:
                # once burning, stays burning
                logits = self._apply_monotone(logits, running_fire)

            outs.append(logits)
            pred_fire = (torch.sigmoid(logits) > 0.5).float()

            if d < n_days:
                if teacher_fire_masks is not None:
                    next_fire = teacher_fire_masks[:, d - 1].to(device).float()
                else:
                    next_fire = pred_fire
                cur_x = cur_x.clone()
                cur_x[:, 0] = next_fire
                running_fire = torch.maximum(running_fire, next_fire)

        return torch.stack(outs, dim=1)
