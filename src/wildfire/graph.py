"""Static grid-graph topology + per-batch physics edge features.

The 64x64 grid has identical topology across every event, so we precompute edge_index
once and reuse. Edge features (wind-edge alignment, slope, NDVI continuity) depend on
the per-event input features and are computed on the fly.

Feature index convention matches data.ndws.FEATURE_ORDER:
    0 PrevFireMask, 1 elevation, 2 th(wind dir deg), 3 vs(wind speed),
    4 tmmn, 5 tmmx, 6 sph, 7 pr, 8 pdsi, 9 NDVI, 10 erc, 11 population
"""
from __future__ import annotations

import math

import torch

H, W = 64, 64
N = H * W  # 4096

# Indices for physics-relevant features (after normalization, but order unchanged)
F_FIRE = 0
F_ELEV = 1
F_WIND_DIR = 2
F_WIND_SPD = 3
F_NDVI = 9


def build_grid_edges(connectivity: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
    """Build edge_index (2, E) and unit edge directions (E, 2) for a 64x64 grid.

    Edge direction is the unit vector from src to dst in image coordinates
    (y axis points down). Used to compute wind-edge alignment.
    """
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
    norm_mean: torch.Tensor | None = None,  # to denormalize wind dir + elevation if needed
    norm_std: torch.Tensor | None = None,
    uniform: bool = False,
) -> torch.Tensor:
    """Per-event edge features (B, E, 3): [wind alignment, slope, NDVI continuity].

    x_nodes is normalized (mean/std applied), so to get true wind direction (degrees) and
    elevation differences we de-normalize on the fly. If norm stats are not provided,
    the values are used as-is (slope/wind direction become arbitrary units, which is
    still a valid learnable input).

    If `uniform=True`, returns zeros — used for the edge-encoding ablation.
    """
    B = x_nodes.size(0)
    E = edge_index.size(1)
    if uniform:
        return torch.zeros(B, E, 3, device=x_nodes.device)

    src, dst = edge_index[0], edge_index[1]

    # De-normalize wind direction (degrees) and elevation if stats are given.
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

    # Wind direction convention: 0=N, 90=E (meteorological "from" -> "to" assumed already toward).
    # We want wind velocity vector in image coords (y axis down). Toward-direction:
    rad = wind_deg * (math.pi / 180.0)
    wind_x = torch.sin(rad)        # east
    wind_y = -torch.cos(rad)       # north -> negative y in image coords
    edx = edge_dirs[:, 0].unsqueeze(0)  # (1, E)
    edy = edge_dirs[:, 1].unsqueeze(0)
    align = wind_x * edx + wind_y * edy        # cosine similarity, (B, E)
    slope = elev_dst - elev_src                # downhill negative
    veg_cont = ndvi_src * ndvi_dst             # both vegetated -> high

    # Soft-normalize so each feature has comparable magnitude.
    slope = slope / 100.0   # rough scale: tens of meters between cells
    return torch.stack([align, slope, veg_cont], dim=-1)


def grid_to_nodes(x: torch.Tensor) -> torch.Tensor:
    """(B, F, H, W) -> (B, N, F)."""
    B, Fin, h, w = x.shape
    return x.permute(0, 2, 3, 1).reshape(B, h * w, Fin)


def nodes_to_grid(h: torch.Tensor, height: int = H, width: int = W) -> torch.Tensor:
    """(B, N, F) -> (B, F, H, W)."""
    B, _, Fout = h.shape
    return h.reshape(B, height, width, Fout).permute(0, 3, 1, 2)
