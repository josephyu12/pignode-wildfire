"""Multi-day PI-GNODE rollout (proposal §4.4).

Loads a PI-GNODE checkpoint trained on NDWS (single-day, t -> t+1) and
evaluates it on TS-SatFire's multi-day fire progressions at horizons
1, 2, 3 days. Reports AUC-PR / AUC-ROC / CSI / F1 per horizon and writes
experiments/<exp>/rollout_<mode>.json.

Two evaluation modes (proposal §4.4):
    --mode free                 pure ODE rollout, no teacher forcing
    --mode teacher              re-feed ground-truth fire mask at each
                                day boundary (the proposal's fallback)

Resolution mismatch
-------------------
PI-GNODE's graph is fixed to a 64x64 grid (NDWS). TS-SatFire frames are
256x256. We downsample 4x via mean pooling for inputs and max pooling for
fire labels (max preserves "any pixel burning" -> "downsampled pixel
burning"). This is the lightweight cross-dataset sanity test in the
proposal; a paper-grade comparison would retrain at 256x256.

Usage
-----
    python -m wildfire.rollout \
        --ckpt experiments/pignode_uniform_full/best.pt \
        --root data/raw/tssatfire \
        --horizons 1 2 3 --mode free
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data.ndws import get_norm
from .data.tssatfire import TSSatFireDataset, TSSatFirePIGNODEAdapter
from .graph import build_grid_edges
from .metrics import all_metrics
from .models.pignode import PIGNODE


NDWS_H = NDWS_W = 64


def _downsample_x(x: torch.Tensor) -> torch.Tensor:
    """(B, 12, 256, 256) -> (B, 12, 64, 64) via 4x4 mean pool, except channel 0
    (PrevFireMask) which uses max-pool so any burning sub-pixel keeps the cell on."""
    if x.shape[-1] == NDWS_W and x.shape[-2] == NDWS_H:
        return x
    fire = x[:, :1]
    rest = x[:, 1:]
    fire_ds = F.max_pool2d(fire, kernel_size=4)
    rest_ds = F.avg_pool2d(rest, kernel_size=4)
    return torch.cat([fire_ds, rest_ds], dim=1)


def _downsample_y(y: torch.Tensor) -> torch.Tensor:
    """Labels: max-pool to keep any-burning semantics. Handles -1 (ignored).

    Shapes: (..., H, W) -> (..., H/4, W/4)."""
    if y.shape[-1] == NDWS_W and y.shape[-2] == NDWS_H:
        return y
    # Expand to (-1, 1, H, W) for pool, then unsqueeze back.
    leading = y.shape[:-2]
    y4 = y.reshape(-1, 1, y.shape[-2], y.shape[-1]).float()
    y4_pos = (y4 > 0.5).float()
    pooled = F.max_pool2d(y4_pos, kernel_size=4)
    return pooled.view(*leading, NDWS_H, NDWS_W)


def load_pignode(ckpt_path: Path, edge_index, edge_dirs, norm_mean, norm_std,
                 device: str) -> PIGNODE:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    a = ckpt["args"]
    model = PIGNODE(
        edge_index=edge_index, edge_dirs=edge_dirs,
        hidden=a["hidden"], heads=a["heads"],
        ode_layers=a.get("ode_layers", 2),
        t_end=a["t_end"], n_eval_steps=a["n_eval_steps"],
        monotone=a.get("monotone", True),
        uniform_edges=a.get("uniform_edges", False),
        norm_mean=norm_mean, norm_std=norm_std,
        solver=a.get("solver", "rk4"), adjoint=False,
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def evaluate_rollout(model: PIGNODE, loader: DataLoader, horizons: list[int],
                     mode: str, device: str, max_batches: int | None = None) -> dict:
    max_h = max(horizons)
    per_h_y: dict[int, list[np.ndarray]] = {h: [] for h in horizons}
    per_h_p: dict[int, list[np.ndarray]] = {h: [] for h in horizons}

    for bi, (xs, ys) in enumerate(loader):
        # xs: (B, T, 12, H, W). T must be >= max_h.
        if xs.shape[1] < max_h:
            raise RuntimeError(
                f"TS-SatFire time-series too short (T={xs.shape[1]}) for horizon "
                f"{max_h}. Re-preprocess with -ts >= {max_h + 1}."
            )

        xs = xs.to(device, non_blocking=True)
        ys = ys.to(device, non_blocking=True)
        x0 = _downsample_x(xs[:, 0])               # (B, 12, 64, 64)
        ys_ds = _downsample_y(ys)                  # (B, T, 64, 64)

        if mode == "teacher":
            tf = (ys_ds[:, : max_h - 1] > 0.5).float()
        else:
            tf = None

        with torch.no_grad():
            logits = model.forward_rollout(x0, n_days=max_h, teacher_fire_masks=tf)
            probs = torch.sigmoid(logits).cpu().numpy()      # (B, max_h, 64, 64)

        ys_np = ys_ds.cpu().numpy()
        for h in horizons:
            per_h_y[h].append(ys_np[:, h - 1])
            per_h_p[h].append(probs[:, h - 1])

        if max_batches and bi + 1 >= max_batches:
            break

    return {
        f"day_{h}": all_metrics(np.concatenate(per_h_y[h], axis=0),
                                np.concatenate(per_h_p[h], axis=0))
        for h in horizons
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, type=Path)
    p.add_argument("--root", default="data/raw/tssatfire", type=Path)
    p.add_argument("--horizons", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--mode", choices=["free", "teacher"], default="free")
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-batches", type=int, default=None)
    p.add_argument("--subset", type=int, default=None,
                   help="evaluate on at most N samples (deterministic stride)")
    p.add_argument("--out", default=None,
                   help="output json path (default: <ckpt parent>/rollout_<mode>.json)")
    args = p.parse_args()

    if not args.ckpt.exists():
        raise SystemExit(f"checkpoint not found: {args.ckpt}")

    base = TSSatFireDataset("test", root=args.root, subset=args.subset)
    ds = TSSatFirePIGNODEAdapter(base, multiday=True)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    ei, ed = build_grid_edges()
    ei = ei.to(args.device); ed = ed.to(args.device)
    ns = get_norm()
    model = load_pignode(args.ckpt, ei, ed, ns["mean"], ns["std"], args.device)

    results = evaluate_rollout(model, loader, args.horizons, args.mode, args.device,
                               max_batches=args.max_batches)

    out_path = Path(args.out) if args.out else args.ckpt.parent / f"rollout_{args.mode}.json"
    with open(out_path, "w") as f:
        json.dump({"args": {k: str(v) for k, v in vars(args).items()},
                   "results": results}, f, indent=2)
    print(json.dumps(results, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
