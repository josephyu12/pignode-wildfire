"""Focal/masked BCE for the imbalanced labels, plus the Frobenius dynamics
regularizer and a soft burn-monotonicity penalty."""
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
    """mean_{burning} ReLU(1 - sigmoid(z))  — soft burn-irreversibility.

    Differentiable surrogate for "p_pred >= p_init on cells already burning".
    Unlike the hard floor, this still gives gradient on burning cells.
    """
    p = torch.sigmoid(logits)
    burning = (prev_fire > threshold).float()
    n_burning = burning.sum().clamp_min(1.0)
    deficit = F.relu(1.0 - p) * burning
    return deficit.sum() / n_burning


def frobenius_dynamics_penalty(norms: list[torch.Tensor] | torch.Tensor) -> torch.Tensor:
    """Mean ||dh/dt||^2 along the trajectory. Keeps dynamics from getting
    stiff (which blows up adaptive solver NFE)."""
    if isinstance(norms, list):
        if not norms:
            return torch.zeros((), requires_grad=False)
        norms = torch.stack(norms)
    return norms.mean()
