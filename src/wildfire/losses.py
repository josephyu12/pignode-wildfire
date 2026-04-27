"""Losses for severe class imbalance: focal + masked BCE."""
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
