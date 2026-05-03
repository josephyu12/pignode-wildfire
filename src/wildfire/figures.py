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
    if not EXP_ROOT.exists():
        return runs
    for p in sorted(EXP_ROOT.iterdir()):
        if not p.is_dir() or p.name.startswith("_"):
            continue
        m = p / "metrics.json"
        if not m.exists():
            continue
        try:
            with open(m) as f:
                runs.append((p.name, json.load(f)))
        except (json.JSONDecodeError, OSError) as e:
            print(f"  skipping corrupt {m}: {e}")
    return runs


def make_results_table():
    runs = _load_runs()
    # Display names + ordering for the paper table.
    display = {
        "lr":                   ("Logistic Regression", "Trad."),
        "rf":                   ("Random Forest",       "Trad."),
        "convae":               ("Conv. Autoencoder",   "CNN"),
        "gcn":                  ("GCN",                 "GNN"),
        "sage":                 ("GraphSAGE",           "GNN"),
        "gat":                  ("GAT (edge-attn)",     "GNN"),
        "pignode_uniform_full": ("PI-GNODE (ours, full data)", "Ours"),
        "pignode":              ("PI-GNODE (ours, 5K subset)", "Ours"),
        "pignode_full":         ("  + physics edges",  "Ablation"),
        "pignode_no_mono":      ("  - monotonicity",   "Ablation"),
    }
    rows = []
    for short, (label, group) in display.items():
        match = next((r for n, r in runs if n == short), None)
        if match is None:
            continue
        t = match["test"]
        rows.append({
            "model": label, "group": group,
            "AUC-PR": t["auc_pr"], "AUC-ROC": t["auc_roc"],
            "CSI": t["csi"], "F1": t["f1"],
        })
    df = pd.DataFrame(rows).set_index("model")
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "results_table.csv")
    with open(OUT / "results_table.tex", "w") as f:
        f.write(df.drop(columns=["group"]).round(3).to_latex(float_format="%.3f"))
    print(df.drop(columns=["group"]).round(3).to_string())
    return df


def plot_curves():
    runs = _load_runs()
    # Only show models with multi-epoch curves (skip LR/RF + duplicates).
    keep = {"convae", "gcn", "sage", "gat",
            "pignode", "pignode_no_mono", "pignode_full", "pignode_uniform_full"}
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, r in runs:
        if name not in keep:
            continue
        h = r["history"]
        if not h:
            continue
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
    """Ablation: main vs +physics-edges vs -monotonicity. All on h96+ode_layers=2.
    Note: the +physics-edges variant is on full data (5 ep), -monotonicity is on
    5K (8 ep) due to compute. Trends are robust to this -- both make it worse.
    """
    runs = dict(_load_runs())
    cells = [
        ("PI-GNODE (main)",     "pignode_uniform_full"),
        ("+ physics edges",     "pignode_full"),
        ("− monotonicity",      "pignode_no_mono"),
    ]
    have = [(label, k) for label, k in cells if k in runs]
    if len(have) < 2:
        print(f"  ablation skipped (have {list(runs)})")
        return
    metrics = ["auc_pr", "csi", "f1"]
    metric_labels = {"auc_pr": "AUC-PR", "csi": "CSI", "f1": "F1"}
    x = np.arange(len(metrics))
    width = 0.8 / len(have)
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#1f77b4", "#ff7f0e", "#d62728"]
    for i, (label, k) in enumerate(have):
        vals = [runs[k]["test"][m] for m in metrics]
        bars = ax.bar(x + i * width, vals, width, label=label, color=colors[i % len(colors)])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width()/2, v + 0.005, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x + width * (len(have) - 1) / 2)
    ax.set_xticklabels([metric_labels[m] for m in metrics])
    ax.set_ylabel("test score")
    ax.set_title("PI-GNODE ablation")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, max(runs[k]["test"]["f1"] for _, k in have) * 1.2)
    fig.tight_layout()
    fig.savefig(OUT / "ablation.png", dpi=140)
    plt.close(fig)


def plot_qualitative(model_dirname: str = "pignode", n_examples: int = 4,
                     device: str | None = None):
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
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


def plot_region_split():
    """2x2 cross-region generalization matrix (replaces TS-SatFire claim).

    Reads experiments/pignode_{high,low}_elev/eval_test_{high,low}_elev.json and
    builds a heatmap: rows = trained_on region, cols = evaluated_on region.
    Diagonal = in-domain; off-diagonal = generalization gap.
    """
    cells = {}
    for trained in ("high_elev", "low_elev"):
        for evald in ("high_elev", "low_elev"):
            path = EXP_ROOT / f"pignode_{trained}" / f"eval_test_{evald}.json"
            if not path.exists():
                print(f"  skip region figure: missing {path}")
                return
            with open(path) as f:
                cells[(trained, evald)] = json.load(f)["metrics"]

    metrics = ["auc_pr", "csi"]
    labels = {"high_elev": "High elevation\n(mountainous)",
              "low_elev":  "Low elevation\n(lowland)"}
    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics) + 1, 4))
    if len(metrics) == 1:
        axes = [axes]
    for ax, metric in zip(axes, metrics):
        mat = np.array([
            [cells[("high_elev", "high_elev")][metric],
             cells[("high_elev", "low_elev")][metric]],
            [cells[("low_elev",  "high_elev")][metric],
             cells[("low_elev",  "low_elev")][metric]],
        ])
        im = ax.imshow(mat, cmap="viridis", vmin=0)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{mat[i,j]:.3f}",
                        ha="center", va="center",
                        color="white" if mat[i,j] < mat.max() * 0.6 else "black",
                        fontsize=12)
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels([labels["high_elev"], labels["low_elev"]], fontsize=8)
        ax.set_yticklabels([labels["high_elev"], labels["low_elev"]], fontsize=8)
        ax.set_xlabel("Evaluated on")
        ax.set_ylabel("Trained on")
        ax.set_title(f"Test {metric.upper().replace('_', '-')}")
        plt.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Cross-region generalization (NDWS elevation split)")
    fig.tight_layout()
    fig.savefig(OUT / "region_split.png", dpi=140)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # Each plotter is wrapped: a single corrupt experiments/<name>/metrics.json
    # should not block the others. We always finish writing the results table
    # so even a one-experiment run produces a usable artifact.
    print("Results table:")
    try:
        make_results_table()
    except Exception as e:
        print(f"  results table failed: {e}")
    print("\nTraining curves...")
    try:
        plot_curves()
    except Exception as e:
        print(f"  curves failed: {e}")
    print("Ablation chart...")
    try:
        plot_ablation()
    except Exception as e:
        print(f"  ablation chart failed: {e}")
    print("Region-split heatmap...")
    try:
        plot_region_split()
    except Exception as e:
        print(f"  region figure failed: {e}")
    print("Qualitative panels...")
    # Use the strongest available pignode checkpoint for the qualitative figure.
    candidates = [
        "pignode_uniform_full", "pignode", "pignode_high_elev", "pignode_low_elev",
    ]
    chosen = next((c for c in candidates if (EXP_ROOT / c / "best.pt").exists()), None)
    if chosen is None:
        print("  no pignode checkpoint found, skipping qualitative")
    else:
        try:
            plot_qualitative(chosen)
        except Exception as e:
            print(f"  qualitative figure failed: {e}")
    print(f"\nAll figures written to {OUT}/")


if __name__ == "__main__":
    main()
