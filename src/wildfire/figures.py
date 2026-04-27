"""Generate paper figures from trained experiments/.

Outputs everything to experiments/_figures/:
  - results_table.csv   (test metrics for every model)
  - results_table.tex   (LaTeX-formatted)
  - curves.png          (training curves: AUC-PR vs epoch, all models)
  - ablation.png        (bar chart of monotonicity / edge-encoding ablation)
  - qualitative_<i>.png (PI-GNODE prediction vs ground truth for a few held-out events)

Usage:
    python -m wildfire.figures
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

EXP_ROOT = Path("experiments")
OUT = EXP_ROOT / "_figures"


def _load_runs() -> list[tuple[str, dict]]:
    runs = []
    for p in sorted(EXP_ROOT.iterdir()):
        if not p.is_dir() or p.name.startswith("_"):
            continue
        m = p / "metrics.json"
        if not m.exists():
            continue
        with open(m) as f:
            runs.append((p.name, json.load(f)))
    return runs


def make_results_table():
    runs = _load_runs()
    rows = []
    for name, r in runs:
        t = r["test"]
        rows.append({
            "model": name,
            "AUC-PR": t["auc_pr"],
            "AUC-ROC": t["auc_roc"],
            "CSI": t["csi"],
            "F1": t["f1"],
            "thresh": t["threshold"],
        })
    df = pd.DataFrame(rows).set_index("model").sort_values("AUC-PR", ascending=False)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "results_table.csv")
    # LaTeX table
    with open(OUT / "results_table.tex", "w") as f:
        f.write(df.round(3).to_latex(float_format="%.3f"))
    print(df.round(3).to_string())
    return df


def plot_curves():
    runs = _load_runs()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, r in runs:
        h = r["history"]
        epochs = [e["epoch"] for e in h]
        aucs = [e["auc_pr"] for e in h]
        ax.plot(epochs, aucs, marker="o", label=name)
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation AUC-PR")
    ax.set_title("Training curves")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "curves.png", dpi=140)
    plt.close(fig)


def plot_ablation():
    runs = dict(_load_runs())
    keys = ["pignode", "pignode_no_mono", "pignode_uniform"]
    have = [k for k in keys if k in runs]
    if len(have) < 2:
        print(f"  ablation skipped (need {keys}, have {list(runs)})")
        return
    labels = {"pignode": "Full PI-GNODE", "pignode_no_mono": "−monotonicity",
              "pignode_uniform": "−physics edges"}
    metrics = ["auc_pr", "csi", "f1"]
    metric_labels = {"auc_pr": "AUC-PR", "csi": "CSI", "f1": "F1"}
    x = np.arange(len(metrics))
    width = 0.8 / len(have)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for i, k in enumerate(have):
        vals = [runs[k]["test"][m] for m in metrics]
        ax.bar(x + i * width, vals, width, label=labels[k])
    ax.set_xticks(x + width * (len(have) - 1) / 2)
    ax.set_xticklabels([metric_labels[m] for m in metrics])
    ax.set_ylabel("test score")
    ax.set_title("PI-GNODE ablation")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "ablation.png", dpi=140)
    plt.close(fig)


def plot_qualitative(model_dirname: str = "pignode", n_examples: int = 4, device: str = "mps"):
    """For a trained PI-GNODE checkpoint, plot input fire / pred / ground truth side-by-side
    for the highest-fire-content events in the test split."""
    from .data.ndws import NDWSDataset
    from .graph import build_grid_edges
    from .models.pignode import PIGNODE

    ckpt_path = EXP_ROOT / model_dirname / "best.pt"
    if not ckpt_path.exists():
        print(f"  {ckpt_path} not found, skipping qualitative")
        return
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    args = ckpt["args"]

    test = NDWSDataset("test", augment=False)
    # Pick events with most fire pixels for visual interest
    fire_count = []
    for i in range(min(200, len(test))):
        _, y = test[i]
        fire_count.append((int((y == 1).sum()), i))
    fire_count.sort(reverse=True)
    pick = [i for _, i in fire_count[:n_examples]]

    ei, ed = build_grid_edges()
    ei = ei.to(device); ed = ed.to(device)
    model = PIGNODE(
        ei, ed, hidden=args["hidden"], heads=args["heads"],
        t_end=args["t_end"], n_eval_steps=args["n_eval_steps"],
        monotone=args.get("monotone", True),
        uniform_edges=args.get("uniform_edges", False),
        norm_mean=test.mean.flatten(), norm_std=test.std.flatten(),
        solver=args.get("solver", "rk4"), adjoint=False,
    ).to(device).eval()
    model.load_state_dict(ckpt["state_dict"])

    fig, axes = plt.subplots(n_examples, 4, figsize=(11, 2.5 * n_examples))
    if n_examples == 1:
        axes = axes[None, :]
    for row, idx in enumerate(pick):
        x, y = test[idx]
        with torch.no_grad():
            prob = torch.sigmoid(model(x.unsqueeze(0).to(device))).cpu().numpy()[0]
        prev_fire = x[0].numpy()       # PrevFireMask (normalized but binary)
        for col, (img, title, cmap) in enumerate([
            (prev_fire, "Prev fire (t)", "Reds"),
            (prob, "Pred prob (t+1)", "viridis"),
            (y.numpy(), "True (t+1)", "Reds"),
            ((prob >= 0.5).astype(np.float32) - y.numpy(),
             "Pred − truth", "RdBu_r"),
        ]):
            ax = axes[row, col]
            vmin, vmax = (-1, 1) if col == 3 else (0, 1)
            ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_xticks([]); ax.set_yticks([])
            if row == 0:
                ax.set_title(title, fontsize=10)
        axes[row, 0].set_ylabel(f"event {idx}", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / f"qualitative_{model_dirname}.png", dpi=140)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("Results table:")
    make_results_table()
    print("\nTraining curves...")
    plot_curves()
    print("Ablation chart...")
    plot_ablation()
    print("Qualitative panels...")
    plot_qualitative("pignode")
    print(f"\nAll figures written to {OUT}/")


if __name__ == "__main__":
    main()
