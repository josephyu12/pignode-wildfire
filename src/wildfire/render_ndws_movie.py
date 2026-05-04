"""Render NDWS sample-montage MP4s for the project video.

Each frame in the movie is a different test fire shown as a 1x3 panel:
    [Day t prev-fire mask] | [Day t+1 ground truth] | [PI-GNODE P(burn) at t+1]

This is a single-day prediction reel (NDWS is t -> t+1), not a multi-day rollout.
For multi-day rollouts you need TS-SatFire; see `render_rollout.py`.

Two modes:
    --mode single    one model, montage of N samples       (default)
    --mode versus    two models on the same N samples,     (cross-region story)
                     stacked top-vs-bottom in each frame

Usage
-----
    # main reel: 30 NDWS test fires, PI-GNODE main checkpoint
    python -m wildfire.render_ndws_movie \\
        --ckpt experiments/pignode_uniform_full/best.pt \\
        --out experiments/_movies/ndws_montage.mp4 \\
        --n-samples 30 --fps 2

    # cross-region versus: low-elev model vs high-elev model on low-elev fires
    python -m wildfire.render_ndws_movie --mode versus \\
        --ckpt experiments/pignode_low_elev/best.pt  \\
        --ckpt-b experiments/pignode_high_elev/best.pt \\
        --region low_elev \\
        --out experiments/_movies/cross_region_low.mp4 \\
        --n-samples 24 --fps 2
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap

from .data.ndws import NDWSDataset, get_norm
from .graph import build_grid_edges
from .rollout import load_pignode


FIRE_CMAP = ListedColormap([(0.05, 0.05, 0.08, 1.0), (1.0, 0.4, 0.05, 1.0)])
PREV_CONTOUR_COLOR = "cyan"
GT_CONTOUR_COLOR = "lime"


def _normalize(x_raw: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (x_raw - mean.reshape(-1, 1, 1)) / std.reshape(-1, 1, 1)


def _pick_samples(ds: NDWSDataset, n: int, min_fire_pixels: int) -> list[int]:
    """Return up to n test indices whose day-0 fire mask has >= min_fire_pixels burning."""
    chosen: list[int] = []
    for i in range(len(ds)):
        x_raw, _ = ds._load(i)                      # raw, un-normalized
        prev_fire = (x_raw[0] > 0.5).sum()
        if prev_fire >= min_fire_pixels:
            chosen.append(i)
            if len(chosen) >= n:
                break
    return chosen


def _draw_mask_contour(ax, mask, color: str, linewidth: float = 0.7) -> None:
    """Draw a binary-mask boundary if a boundary exists in-frame."""
    m = np.asarray(mask)
    if np.any(m > 0.5) and np.any(m <= 0.5):
        ax.contour(m, levels=[0.5], colors=color, linewidths=linewidth)


def _panel_single(ax_row, prev_fire, gt, prob, title_prefix: str):
    ax_row[0].imshow(prev_fire, cmap=FIRE_CMAP, vmin=0, vmax=1, interpolation="nearest")
    ax_row[0].set_title(f"{title_prefix}day t  (prev fire)", fontsize=10)

    ax_row[1].imshow(gt, cmap=FIRE_CMAP, vmin=0, vmax=1, interpolation="nearest")
    _draw_mask_contour(ax_row[1], prev_fire, PREV_CONTOUR_COLOR, linewidth=0.6)
    ax_row[1].set_title(f"{title_prefix}day t+1 GT", fontsize=10)

    im = ax_row[2].imshow(prob, cmap="inferno", vmin=0, vmax=1, interpolation="nearest")
    _draw_mask_contour(ax_row[2], prev_fire, PREV_CONTOUR_COLOR, linewidth=0.6)
    _draw_mask_contour(ax_row[2], gt, GT_CONTOUR_COLOR, linewidth=0.8)
    ax_row[2].set_title(f"{title_prefix}PI-GNODE  P(burn) + GT", fontsize=10)
    for ax in ax_row:
        ax.set_xticks([]); ax.set_yticks([])
    return im


def _render_frame_single(out_path: Path, sample_idx: int,
                         prev_fire, gt, prob) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.4), constrained_layout=True)
    fig.suptitle(f"NDWS test sample {sample_idx:04d}", fontsize=11)
    im = _panel_single(axes, prev_fire, gt, prob, title_prefix="")
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.02)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _render_frame_versus(out_path: Path, sample_idx: int,
                         prev_fire, gt, prob_a, prob_b,
                         label_a: str, label_b: str) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(9.5, 6.4), constrained_layout=True)
    fig.suptitle(f"NDWS test sample {sample_idx:04d}", fontsize=11)
    im = _panel_single(axes[0], prev_fire, gt, prob_a, title_prefix="")
    fig.colorbar(im, ax=axes[0, 2], fraction=0.046, pad=0.02)
    axes[0, 0].text(0.02, 0.98, label_a, transform=axes[0, 0].transAxes,
                    color="white", fontsize=9, ha="left", va="top",
                    bbox={"facecolor": "black", "alpha": 0.55, "pad": 2, "edgecolor": "none"})
    im = _panel_single(axes[1], prev_fire, gt, prob_b, title_prefix="")
    fig.colorbar(im, ax=axes[1, 2], fraction=0.046, pad=0.02)
    axes[1, 0].text(0.02, 0.98, label_b, transform=axes[1, 0].transAxes,
                    color="white", fontsize=9, ha="left", va="top",
                    bbox={"facecolor": "black", "alpha": 0.55, "pad": 2, "edgecolor": "none"})
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _stitch_mp4(frames_dir: Path, mp4_path: Path, fps: int) -> None:
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg not found on PATH; cannot stitch MP4")
    mp4_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-framerate", str(fps),
        "-i", str(frames_dir / "frame_%04d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        str(mp4_path),
    ]
    subprocess.run(cmd, check=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, type=Path)
    p.add_argument("--ckpt-b", type=Path, default=None,
                   help="(versus mode) second checkpoint to compare against")
    p.add_argument("--label-a", default="model A")
    p.add_argument("--label-b", default="model B")
    p.add_argument("--mode", choices=["single", "versus"], default="single")
    p.add_argument("--region", choices=["all", "high_elev", "low_elev"], default="all")
    p.add_argument("--ndws-root", default="data/raw/ndws", type=Path)
    p.add_argument("--out", required=True, type=Path, help="output mp4 path")
    p.add_argument("--n-samples", type=int, default=30)
    p.add_argument("--min-fire-pixels", type=int, default=50,
                   help="skip samples whose day-0 fire mask has fewer than N burning cells")
    p.add_argument("--indices", type=int, nargs="+", default=None,
                   help="explicit sample indices to render (overrides --n-samples / filter)")
    p.add_argument("--fps", type=int, default=2)
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = p.parse_args()

    if args.mode == "versus" and args.ckpt_b is None:
        raise SystemExit("--mode versus requires --ckpt-b")

    region = None if args.region == "all" else args.region
    ds = NDWSDataset("test", root=args.ndws_root, normalize=False,
                     in_memory=False, region=region)
    print(f"NDWS test ({region or 'all'}): {len(ds)} samples")

    if args.indices is not None:
        indices = [i for i in args.indices if 0 <= i < len(ds)]
    else:
        indices = _pick_samples(ds, args.n_samples, args.min_fire_pixels)
    if not indices:
        raise SystemExit("no samples passed the fire-pixel filter; try lowering --min-fire-pixels")
    print(f"rendering {len(indices)} frames -> {args.out}")

    ns = get_norm()
    mean, std = ns["mean"], ns["std"]
    ei, ed = build_grid_edges()
    ei = ei.to(args.device); ed = ed.to(args.device)
    model_a = load_pignode(args.ckpt, ei, ed, mean, std, args.device)
    model_b = load_pignode(args.ckpt_b, ei, ed, mean, std, args.device) if args.mode == "versus" else None

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        for k, idx in enumerate(indices):
            x_raw, y_raw = ds._load(idx)
            prev_fire = (x_raw[0] > 0.5).astype(np.float32)
            gt = np.where(y_raw > 0.5, 1.0, 0.0).astype(np.float32)

            x_n = _normalize(x_raw, mean, std)
            x_t = torch.from_numpy(x_n).unsqueeze(0).to(args.device)

            with torch.no_grad():
                logits_a = model_a(x_t)
                if isinstance(logits_a, tuple):
                    logits_a = logits_a[0]
                prob_a = torch.sigmoid(logits_a).squeeze(0).cpu().numpy()
                if model_b is not None:
                    logits_b = model_b(x_t)
                    if isinstance(logits_b, tuple):
                        logits_b = logits_b[0]
                    prob_b = torch.sigmoid(logits_b).squeeze(0).cpu().numpy()

            frame_path = tmp_dir / f"frame_{k:04d}.png"
            if args.mode == "single":
                _render_frame_single(frame_path, idx, prev_fire, gt, prob_a)
            else:
                _render_frame_versus(frame_path, idx, prev_fire, gt, prob_a, prob_b,
                                     args.label_a, args.label_b)

        _stitch_mp4(tmp_dir, args.out, fps=args.fps)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
