"""Evaluate a trained checkpoint on an NDWS region without retraining.

This is the cross-region generalization measurement: a model trained on
high-elevation events is evaluated on the held-out low-elevation test split
(and vice versa). Reports the standard metric panel and writes
experiments/<exp>/eval_<region>.json.

Usage
-----
    python -m wildfire.eval_region \\
        --ckpt experiments/pignode_high_elev/best.pt \\
        --region low_elev
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data.ndws import NDWSDataset, get_norm
from .graph import build_grid_edges
from .metrics import all_metrics
from .models.baselines import ConvAE
from .models.gnns import (
    GridGAT,
    GridGATEdge,
    GridGCN,
    GridGCNEdge,
    GridSAGE,
    GridSAGEEdge,
)
from .models.pignode import PIGNODE


def _build_model(model_name: str, args_ckpt: dict, edge_index, edge_dirs,
                 norm_mean, norm_std):
    name = model_name.lower()
    hid = args_ckpt.get("hidden", 64)
    n_layers = args_ckpt.get("n_layers", 3)
    if name == "convae":
        return ConvAE(in_ch=12, base=hid // 2)
    if name == "gcn":
        return GridGCN(edge_index, hid=hid, n_layers=n_layers)
    if name == "sage":
        return GridSAGE(edge_index, hid=hid, n_layers=n_layers)
    if name == "gat":
        return GridGAT(edge_index, hid=hid, n_layers=n_layers)
    if name == "gcn_edge":
        return GridGCNEdge(edge_index, edge_dirs, hid=hid, n_layers=n_layers,
                           uniform_edges=args_ckpt.get("uniform_edges", False),
                           norm_mean=norm_mean, norm_std=norm_std)
    if name == "sage_edge":
        return GridSAGEEdge(edge_index, edge_dirs, hid=hid, n_layers=n_layers,
                            uniform_edges=args_ckpt.get("uniform_edges", False),
                            norm_mean=norm_mean, norm_std=norm_std)
    if name == "gat_edge":
        return GridGATEdge(edge_index, edge_dirs, hid=hid, n_layers=n_layers,
                           heads=args_ckpt.get("heads", 4),
                           uniform_edges=args_ckpt.get("uniform_edges", False),
                           norm_mean=norm_mean, norm_std=norm_std)
    if name == "pignode":
        return PIGNODE(
            edge_index=edge_index, edge_dirs=edge_dirs,
            hidden=hid, heads=args_ckpt.get("heads", 4),
            ode_layers=args_ckpt.get("ode_layers", 2),
            t_end=args_ckpt["t_end"], n_eval_steps=args_ckpt["n_eval_steps"],
            monotone=args_ckpt.get("monotone", True),
            uniform_edges=args_ckpt.get("uniform_edges", False),
            norm_mean=norm_mean, norm_std=norm_std,
            solver=args_ckpt.get("solver", "rk4"), adjoint=False,
        )
    raise ValueError(f"unknown model: {name!r}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, type=Path)
    p.add_argument("--region", required=True,
                   choices=["all", "high_elev", "low_elev"])
    p.add_argument("--split", default="test", choices=["train", "eval", "test"])
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    if not args.ckpt.exists():
        raise SystemExit(f"checkpoint not found: {args.ckpt}")

    ckpt = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    a = ckpt["args"]
    model_name = a["model"]
    print(f"Loading {model_name} from {args.ckpt}")
    print(f"  trained on region: {a.get('region', 'all')!r}")
    print(f"  evaluating on region: {args.region!r} ({args.split} split)")

    region = args.region if args.region != "all" else None
    ds = NDWSDataset(
        args.split, augment=False, in_memory=True,
        drop_feature_group=a.get("drop_feature_group") or None,
        region=region,
    )
    print(f"  {len(ds)} events in this region/split")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    ns = get_norm()
    edge_index, edge_dirs = build_grid_edges(connectivity=a.get("connectivity", 8))
    edge_index = edge_index.to(args.device); edge_dirs = edge_dirs.to(args.device)

    model = _build_model(model_name, a, edge_index, edge_dirs, ns["mean"], ns["std"])
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(args.device).eval()

    ys, ps = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(args.device, non_blocking=True)
            logits = model(x)
            if isinstance(logits, tuple):
                logits = logits[0]
            ys.append(y.numpy())
            ps.append(torch.sigmoid(logits).cpu().numpy())
    metrics = all_metrics(np.concatenate(ys), np.concatenate(ps))

    out_path = (Path(args.out) if args.out
                else args.ckpt.parent / f"eval_{args.split}_{args.region}.json")
    with open(out_path, "w") as f:
        json.dump({
            "ckpt": str(args.ckpt),
            "trained_on": a.get("region", "all"),
            "evaluated_on": args.region,
            "split": args.split,
            "n_events": len(ds),
            "metrics": metrics,
        }, f, indent=2)
    print(f"\nAUC-PR  {metrics['auc_pr']:.3f}  AUC-ROC {metrics['auc_roc']:.3f}  "
          f"CSI {metrics['csi']:.3f}  F1 {metrics['f1']:.3f}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
