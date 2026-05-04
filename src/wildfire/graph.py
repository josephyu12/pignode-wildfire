"""Grid graph + per-batch edge features (wind alignment, slope, NDVI).

Topology is the same across every event so we build edge_index once.
Feature indices follow data.ndws.FEATURE_ORDER.
"""
from __future__ import annotations

import math

import torch

H, W = 64, 64
N = H * W  # 4096

F_FIRE = 0
F_ELEV = 1
F_WIND_DIR = 2
F_WIND_SPD = 3
F_NDVI = 9


def build_grid_edges(connectivity: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
    """edge_index (2, E) + unit edge directions (E, 2). Direction is src->dst
    in image coords (y axis down) so we can dot with wind."""
    rs, cs = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    src_list, dst_list, dirs = [], [], []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            if connectivity == 4 and dr * dc != 0:
                continue
            nr, nc = rs + dr, cs + dc
            valid = (nr >= 0) & (nr < H) & (nc >= 0) & (nc < W)
            src = (rs * W + cs)[valid]
            dst = (nr * W + nc)[valid]
            n = src.numel()
            src_list.append(src)
            dst_list.append(dst)
            norm = math.sqrt(dr * dr + dc * dc)
            dx = torch.full((n,), dc / norm)
            dy = torch.full((n,), dr / norm)
            dirs.append(torch.stack([dx, dy], dim=1))
    edge_index = torch.stack([torch.cat(src_list), torch.cat(dst_list)], dim=0).long()
    edge_dirs = torch.cat(dirs, dim=0).float()
    return edge_index, edge_dirs


def batched_edge_index(edge_index: torch.Tensor, batch_size: int, n_per: int = N) -> torch.Tensor:
    """Replicate edge_index across batch, offset by n_per per copy. Returns (2, B*E)."""
    offsets = torch.arange(batch_size, device=edge_index.device) * n_per
    return torch.cat([edge_index + o for o in offsets], dim=1)


def compute_edge_features(
    x_nodes: torch.Tensor,            # (B, N, F)
    edge_index: torch.Tensor,         # (2, E)
    edge_dirs: torch.Tensor,          # (E, 2)
    norm_mean: torch.Tensor | None = None,
    norm_std: torch.Tensor | None = None,
    uniform: bool = False,
) -> torch.Tensor:
    """(B, E, 3) edge features: [wind-alignment, slope, NDVI continuity].

    De-normalizes wind/elevation on the fly if norm stats are passed.
    uniform=True -> zeros (ablation).
    """
    B = x_nodes.size(0)
    E = edge_index.size(1)
    if uniform:
        return torch.zeros(B, E, 3, device=x_nodes.device)

    src, dst = edge_index[0], edge_index[1]

    if norm_mean is not None and norm_std is not None:
        nm = norm_mean.to(x_nodes.device)
        ns = norm_std.to(x_nodes.device)
        wind_deg = x_nodes[:, src, F_WIND_DIR] * ns[F_WIND_DIR] + nm[F_WIND_DIR]
        elev_src = x_nodes[:, src, F_ELEV] * ns[F_ELEV] + nm[F_ELEV]
        elev_dst = x_nodes[:, dst, F_ELEV] * ns[F_ELEV] + nm[F_ELEV]
        ndvi_src = x_nodes[:, src, F_NDVI] * ns[F_NDVI] + nm[F_NDVI]
        ndvi_dst = x_nodes[:, dst, F_NDVI] * ns[F_NDVI] + nm[F_NDVI]
    else:
        wind_deg = x_nodes[:, src, F_WIND_DIR]
        elev_src = x_nodes[:, src, F_ELEV]
        elev_dst = x_nodes[:, dst, F_ELEV]
        ndvi_src = x_nodes[:, src, F_NDVI]
        ndvi_dst = x_nodes[:, dst, F_NDVI]

    # 0=N, 90=E. Want the wind velocity in image coords (y axis down).
    rad = wind_deg * (math.pi / 180.0)
    wind_x = torch.sin(rad)
    wind_y = -torch.cos(rad)
    edx = edge_dirs[:, 0].unsqueeze(0)
    edy = edge_dirs[:, 1].unsqueeze(0)
    align = wind_x * edx + wind_y * edy
    slope = elev_dst - elev_src
    veg_cont = ndvi_src * ndvi_dst

    slope = slope / 100.0   # ~tens of meters per cell
    return torch.stack([align, slope, veg_cont], dim=-1)


def grid_to_nodes(x: torch.Tensor) -> torch.Tensor:
    B, Fin, h, w = x.shape
    return x.permute(0, 2, 3, 1).reshape(B, h * w, Fin)


def nodes_to_grid(h: torch.Tensor, height: int = H, width: int = W) -> torch.Tensor:
    B, _, Fout = h.shape
    return h.reshape(B, height, width, Fout).permute(0, 3, 1, 2)
