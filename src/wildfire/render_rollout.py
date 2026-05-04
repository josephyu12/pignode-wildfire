"""Render PI-GNODE multi-day rollouts as side-by-side MP4s for the project video.

For each selected TS-SatFire test sample, this:
  1. runs PI-GNODE forward for `--n-days` days
  2. saves per-day PNG frames showing [GT fire] | [predicted prob] | [predicted >0.5]
  3. stitches frames into an MP4 via ffmpeg

Usage
-----
    python -m wildfire.render_rollout \
        --ckpt experiments/pignode_uniform_full/best.pt \
        --root data/raw/tssatfire \
        --out-dir experiments/_movies \
        --n-samples 6 --n-days 3 --fps 1
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap
from torch.utils.data import DataLoader

from .data.ndws import get_norm
from .data.tssatfire import TSSatFireDataset, TSSatFirePIGNODEAdapter
from .graph import build_grid_edges
from .models.pignode import PIGNODE
from .rollout import _downsample_x, _downsample_y, load_pignode


FIRE_CMAP = ListedColormap([(0.05, 0.05, 0.08, 1.0), (1.0, 0.4, 0.05, 1.0)])
PREV_CONTOUR_COLOR = "cyan"
GT_CONTOUR_COLOR = "lime"


def _draw_mask_contour(ax, mask, color: str, linewidth: float = 0.7) -> None:
    """Draw a binary-mask boundary if a boundary exists in-frame."""
    m = np.asarray(mask)
    if np.any(m > 0.5) and np.any(m <= 0.5):
        ax.contour(m, levels=[0.5], colors=color, linewidths=linewidth)


def _render_frame(out_path: Path, day: int, n_days: int,
                  init_fire: np.ndarray, gt: np.ndarray,
                  prob: np.ndarray, sample_label: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.4), constrained_layout=True)
    fig.suptitle(f"{sample_label}  ·  Day {day} / {n_days}", fontsize=11)

    # Col 1: GT fire mask, with day-0 ignition outlined for context.
    ax = axes[0]
    ax.imshow(gt, cmap=FIRE_CMAP, vmin=0, vmax=1, interpolation="nearest")
    _draw_mask_contour(ax, init_fire, PREV_CONTOUR_COLOR, linewidth=0.6)
    ax.set_title("ground truth", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])

    # Col 2: predicted probability heatmap.
    ax = axes[1]
    im = ax.imshow(prob, cmap="inferno", vmin=0, vmax=1, interpolation="nearest")
    _draw_mask_contour(ax, init_fire, PREV_CONTOUR_COLOR, linewidth=0.6)
    _draw_mask_contour(ax, gt, GT_CONTOUR_COLOR, linewidth=0.8)
    ax.set_title("PI-GNODE  P(burn) + GT", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    # Col 3: thresholded prediction.
    ax = axes[2]
    ax.imshow((prob > 0.5).astype(np.float32), cmap=FIRE_CMAP, vmin=0, vmax=1,
              interpolation="nearest")
    _draw_mask_contour(ax, init_fire, PREV_CONTOUR_COLOR, linewidth=0.6)
    _draw_mask_contour(ax, gt, GT_CONTOUR_COLOR, linewidth=0.8)
    ax.set_title("PI-GNODE  > 0.5 + GT", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])

    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _stitch_mp4(frames_dir: Path, mp4_path: Path, fps: int) -> bool:
    if shutil.which("ffmpeg") is None:
        print(f"  ffmpeg not found on PATH — frames saved to {frames_dir}, skipping stitch")
        return False
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-framerate", str(fps),
        "-i", str(frames_dir / "frame_%02d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        str(mp4_path),
    ]
    subprocess.run(cmd, check=True)
    return True


def render_sample(model: PIGNODE, xs: torch.Tensor, ys: torch.Tensor,
                  n_days: int, mode: str, device: str,
                  out_dir: Path, sample_label: str, fps: int) -> Path | None:
    """xs: (T, 12, H, W) raw TS-SatFire snapshot; ys: (T, H, W) labels.
    Writes frames + mp4 under out_dir; returns mp4 path (or None if stitch skipped)."""
    if xs.shape[0] < n_days:
        print(f"  {sample_label}: too short (T={xs.shape[0]} < {n_days}), skipping")
        return None

    x = xs.unsqueeze(0).to(device)                   # (1, T, 12, 256, 256)
    y = ys.unsqueeze(0).to(device)                   # (1, T, 256, 256)
    x0 = _downsample_x(x[:, 0])                      # (1, 12, 64, 64)
    y_ds = _downsample_y(y)                          # (1, T, 64, 64)

    if mode == "teacher":
        tf = (y_ds[:, : n_days - 1] > 0.5).float()
    else:
        tf = None

    with torch.no_grad():
        logits = model.forward_rollout(x0, n_days=n_days, teacher_fire_masks=tf)
        probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()    # (n_days, 64, 64)

    init_fire = (x0[0, 0].cpu().numpy() > 0.5).astype(np.float32)
    gt = (y_ds[0].cpu().numpy() > 0.5).astype(np.float32)         # (T, 64, 64)

    out_dir.mkdir(parents=True, exist_ok=True)
    for d in range(1, n_days + 1):
        _render_frame(
            out_path=out_dir / f"frame_{d:02d}.png",
            day=d, n_days=n_days,
            init_fire=init_fire,
            gt=gt[d - 1],
            prob=probs[d - 1],
            sample_label=sample_label,
        )

    mp4_path = out_dir.with_suffix(".mp4")
    ok = _stitch_mp4(out_dir, mp4_path, fps=fps)
    return mp4_path if ok else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, type=Path)
    p.add_argument("--root", default="data/raw/tssatfire", type=Path)
    p.add_argument("--out-dir", default="experiments/_movies", type=Path)
    p.add_argument("--n-samples", type=int, default=6,
                   help="render the first N test samples (ignored if --indices set)")
    p.add_argument("--indices", type=int, nargs="+", default=None,
                   help="explicit sample indices to render (overrides --n-samples)")
    p.add_argument("--n-days", type=int, default=3)
    p.add_argument("--mode", choices=["free", "teacher"], default="free")
    p.add_argument("--fps", type=int, default=1, help="frames per second in the mp4")
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = p.parse_args()

    if not args.ckpt.exists():
        raise SystemExit(f"checkpoint not found: {args.ckpt}")

    base = TSSatFireDataset("test", root=args.root)
    ds = TSSatFirePIGNODEAdapter(base, multiday=True)

    ei, ed = build_grid_edges()
    ei = ei.to(args.device); ed = ed.to(args.device)
    ns = get_norm()
    model = load_pignode(args.ckpt, ei, ed, ns["mean"], ns["std"], args.device)

    indices = args.indices if args.indices is not None else list(range(min(args.n_samples, len(ds))))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rendered: list[Path] = []
    for i in indices:
        if i >= len(ds):
            print(f"index {i} out of range (dataset has {len(ds)} samples), skipping")
            continue
        xs, ys = ds[i]
        sample_label = f"sample {i:04d}"
        sample_dir = args.out_dir / f"sample_{i:04d}"
        print(f"rendering {sample_label} -> {sample_dir}")
        mp4 = render_sample(model, xs, ys, args.n_days, args.mode, args.device,
                            sample_dir, sample_label, args.fps)
        if mp4 is not None:
            rendered.append(mp4)

    if rendered:
        print("\nrendered MP4s:")
        for m in rendered:
            print(f"  {m}")
    else:
        print("\nno MP4s rendered (frames may still be available in subdirs)")


if __name__ == "__main__":
    main()
