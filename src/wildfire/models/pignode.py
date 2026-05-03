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
    """Time-derivative dh/dt parameterized by stacked edge-conditioned GAT layers.

    PyG's GATConv with edge_dim>0 implements eq. (3) in the proposal:
        α_ij = softmax(LeakyReLU(a^T [Wh_i || Wh_j || W_e e_ij]))

    Stacking >1 GAT layer inside f gives the ODE function genuine multi-hop
    message-passing depth per integration step, which the integration-only
    formulation lacks. Each step thus performs ode_layers hops; integration
    over T steps performs T*ode_layers hops total.

    Optionally records ‖dh/dt‖_F^2 / N at every NFE call for the Frobenius
    dynamics regularizer (proposal §5 #4). Set `record_norm=True` before the
    forward solve and read `dyn_norms` afterwards.
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
        # Scale-down the LAST GAT's projection so dh/dt is small at init -- a
        # standard Neural ODE stability trick ("identity-init" the residual flow).
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
        for i, (gat, norm) in enumerate(zip(self.gats, self.norms)):
            msg = gat(z, self._edge_index, edge_attr=self._edge_attr)
            z = self.act(norm(msg))
        if self.record_norm:
            # Per-element mean of squared norm; cheaper than full Frobenius and
            # invariant to graph size, which keeps the regularizer scale stable.
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
        """Encode nodes, compute edge features, install batched edge_index.

        Returns (h0, ei_batched, edge_attr_flat). Centralizing this makes the
        single-day forward and the multi-day rollout share identical setup.
        """
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
        return h0, ei, edge_attr

    def _integrate(self, h0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Run the ODE solver. `f`'s context (edge_index, edge_attr) must be set first."""
        if self.adjoint:
            return odeint_adjoint(
                self.f, h0, t, method=self.solver,
                rtol=self.rtol, atol=self.atol,
                adjoint_params=tuple(self.f.parameters()),
            )
        return odeint(self.f, h0, t, method=self.solver, rtol=self.rtol, atol=self.atol)

    def _apply_monotone(self, logits: torch.Tensor, init_fire: torch.Tensor) -> torch.Tensor:
        """Eq. (4) hard floor: σ(logits_i) ≥ 1[init_fire_i = 1]. Inference only.

        Applying this during training would kill gradients on burning cells
        (which hold ~99% of the label-1 mass), starving the model of signal.
        We instead use `losses.soft_monotonicity_penalty` during training.
        """
        big = torch.full_like(logits, 6.0)   # σ(6) ≈ 0.998
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
        """Multi-day forecast (proposal §4.4).

        Integrates the ODE over [0, n_days] in unit-day chunks, decoding a fire
        probability map at each integer day. The continuous formulation is the
        whole point: a single longer integration has no autoregressive error
        accumulation, but we still snapshot per-day predictions for evaluation.

        Args:
            x:                   (B, 12, H, W) initial input features at day 0.
            n_days:              number of days to forecast (>= 1).
            teacher_fire_masks:  optional (B, n_days, H, W) ground-truth fire
                masks at day boundaries. If provided, we re-encode at each
                boundary with the true mask substituted into channel 0
                (teacher forcing, proposal §4.4 fallback). Otherwise pure
                continuous integration with the inference monotonicity floor
                applied at each day boundary.

        Returns:
            logits at each day, shape (B, n_days, H, W). Monotonicity floor is
            applied per-day relative to the running burning mask -- once a cell
            is predicted burning at day d, it stays burning at d' > d.
        """
        assert n_days >= 1
        B = x.size(0)
        device = x.device
        outs: list[torch.Tensor] = []

        running_fire = x[:, 0].clone()                  # (B, H, W) burning indicator at day d
        cur_x = x

        for d in range(1, n_days + 1):
            h0, ei, edge_attr = self._prepare_context(cur_x)
            self.f.set_context(ei, edge_attr, record_norm=False)
            t = torch.tensor([0.0, 1.0], device=device, dtype=h0.dtype)
            h_traj = self._integrate(h0, t)
            h_T = h_traj[-1]
            logits = self.head(h_T).view(B, 64, 64)

            if self.monotone:
                # Per-day eq. (4): a cell that was ever burning stays burning.
                logits = self._apply_monotone(logits, running_fire)

            outs.append(logits)
            pred_fire = (torch.sigmoid(logits) > 0.5).float()

            if d < n_days:
                # Re-base the next day. Teacher forcing replaces channel 0 with
                # the ground-truth fire mask if supplied; otherwise we feed back
                # our own thresholded prediction.
                if teacher_fire_masks is not None:
                    next_fire = teacher_fire_masks[:, d - 1].to(device).float()
                else:
                    next_fire = pred_fire
                cur_x = cur_x.clone()
                cur_x[:, 0] = next_fire
                running_fire = torch.maximum(running_fire, next_fire)

        return torch.stack(outs, dim=1)                  # (B, n_days, H, W)
