"""Losses for severe class imbalance: focal + masked BCE.

Also includes the proposal's Frobenius dynamics regularizer (proposal §5
challenge #4) and a soft monotonicity penalty that approximates eq. (4)
during training without killing gradients on burning cells.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def focal_bce_with_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.75,
    gamma: float = 2.0,
    ignore_value: float = -1.0,
) -> torch.Tensor:
    """Binary focal loss; ignores cells where target == ignore_value."""
    valid = target != ignore_value
    if valid.sum() == 0:
        return logits.sum() * 0.0
    z = logits[valid]
    y = target[valid].float()
    p = torch.sigmoid(z)
    p_t = p * y + (1 - p) * (1 - y)
    alpha_t = alpha * y + (1 - alpha) * (1 - y)
    bce = F.binary_cross_entropy_with_logits(z, y, reduction="none")
    loss = alpha_t * (1 - p_t).pow(gamma) * bce
    return loss.mean()


def weighted_bce_with_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    pos_weight: float = 3.0,
    ignore_value: float = -1.0,
) -> torch.Tensor:
    valid = target != ignore_value
    if valid.sum() == 0:
        return logits.sum() * 0.0
    z = logits[valid]
    y = target[valid].float()
    pw = torch.tensor(pos_weight, device=z.device)
    return F.binary_cross_entropy_with_logits(z, y, pos_weight=pw, reduction="mean")


def soft_monotonicity_penalty(
    logits: torch.Tensor,
    prev_fire: torch.Tensor,
    threshold: float = 0.5,
) -> torch.Tensor:
    """Soft analogue of eq. (4) burn irreversibility.

    Penalizes σ(logits_i) < 1[prev_fire_i = 1] for cells that were burning at t.
    A differentiable surrogate for `p̂ ≥ p_init` that, unlike the hard inference
    floor, leaves a learning signal on burning cells. ReLU bottoms out cleanly
    once σ(z) ≥ 1, so cells that already predict "burning" cost zero.

        L_mono = mean_{i : prev_fire_i = 1} ReLU(1 - σ(z_i))

    Shapes:
        logits     : (B, H, W) -- raw logits before any monotonicity floor
        prev_fire  : (B, H, W) -- binary fire mask at t (== x[:, 0])
    """
    p = torch.sigmoid(logits)
    burning = (prev_fire > threshold).float()
    n_burning = burning.sum().clamp_min(1.0)
    deficit = F.relu(1.0 - p) * burning
    return deficit.sum() / n_burning


def frobenius_dynamics_penalty(norms: list[torch.Tensor] | torch.Tensor) -> torch.Tensor:
    """Frobenius-norm regularizer on the ODE derivative f_θ along the trajectory.

    Proposal §5 challenge #4: "regularize dynamics via Frobenius-norm penalties."
    Discourages stiff dynamics (large ||dh/dt||) so the adaptive solver does not
    blow up step counts. We average over all NFE evaluations recorded during the
    forward solve.

    Accepts either a list of per-call scalar tensors or a stacked 1-D tensor.
    Returns 0 if the list is empty (model wasn't an ODE).
    """
    if isinstance(norms, list):
        if not norms:
            return torch.zeros((), requires_grad=False)
        norms = torch.stack(norms)
    return norms.mean()
