"""Training loop for all models.

    python -m wildfire.train --model pignode --epochs 8 --batch-size 16

Output goes to experiments/<exp>/.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset

from .data.ndws import NDWSDataset, get_norm
from .graph import build_grid_edges
from .losses import (
    focal_bce_with_logits,
    frobenius_dynamics_penalty,
    soft_monotonicity_penalty,
)
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


def make_model(name: str, edge_index, edge_dirs, norm_mean, norm_std, **kw):
    name = name.lower()
    hid = kw.get("hidden", 64)
    n_layers = kw.get("n_layers", 3)
    if name == "convae":
        return ConvAE(in_ch=12, base=kw.get("base", 32))
    if name == "gcn":
        return GridGCN(edge_index, hid=hid, n_layers=n_layers)
    if name == "sage":
        return GridSAGE(edge_index, hid=hid, n_layers=n_layers)
    if name == "gat":
        return GridGAT(edge_index, hid=hid, n_layers=n_layers)
    if name == "gcn_edge":
        return GridGCNEdge(
            edge_index, edge_dirs, hid=hid, n_layers=n_layers,
            uniform_edges=kw.get("uniform_edges", False),
            norm_mean=norm_mean, norm_std=norm_std,
        )
    if name == "sage_edge":
        return GridSAGEEdge(
            edge_index, edge_dirs, hid=hid, n_layers=n_layers,
            uniform_edges=kw.get("uniform_edges", False),
            norm_mean=norm_mean, norm_std=norm_std,
        )
    if name == "gat_edge":
        return GridGATEdge(
            edge_index, edge_dirs, hid=hid, n_layers=n_layers,
            heads=kw.get("heads", 4),
            uniform_edges=kw.get("uniform_edges", False),
            norm_mean=norm_mean, norm_std=norm_std,
        )
    if name == "pignode":
        return PIGNODE(
            edge_index=edge_index, edge_dirs=edge_dirs,
            hidden=hid,
            heads=kw.get("heads", 4),
            ode_layers=kw.get("ode_layers", 2),
            t_end=kw.get("t_end", 1.0),
            n_eval_steps=kw.get("n_eval_steps", 2),
            monotone=kw.get("monotone", True),
            uniform_edges=kw.get("uniform_edges", False),
            norm_mean=norm_mean, norm_std=norm_std,
            solver=kw.get("solver", "dopri5"),
            adjoint=kw.get("adjoint", True),
            return_dyn_norms=kw.get("frobenius_weight", 0.0) > 0.0,
        )
    raise ValueError(f"unknown model: {name}")


def evaluate(model, loader, device, max_batches: int | None = None) -> dict:
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            x = x.to(device, non_blocking=True)
            logits = model(x)
            prob = torch.sigmoid(logits).cpu().numpy()
            ys.append(y.numpy()); ps.append(prob)
            if max_batches and i + 1 >= max_batches:
                break
    return all_metrics(np.concatenate(ys), np.concatenate(ps))


def train(args):
    device = torch.device(args.device)
    out = Path("experiments") / args.exp
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "log.txt"

    def log(*a):
        msg = " ".join(str(x) for x in a)
        print(msg, flush=True)
        with open(log_path, "a") as f:
            f.write(msg + "\n")

    log(f"=== {args.exp} | model={args.model} | device={device} ===")

    ns = get_norm()
    norm_mean, norm_std = ns["mean"], ns["std"]

    # in_memory is way faster than slicing zarr on every batch
    log("loading data into memory...")
    t = time.time()
    drop = args.drop_feature_group or None
    if drop:
        log(f"  dropping feature group: {drop!r}")
    region = args.region if args.region != "all" else None
    if region:
        log(f"  region filter: {region!r}")
    train_ds = NDWSDataset("train", augment=True, in_memory=args.in_memory,
                           drop_feature_group=drop, region=region)
    val_ds = NDWSDataset("eval", augment=False, in_memory=args.in_memory,
                         drop_feature_group=drop, region=region)
    test_ds = NDWSDataset("test", augment=False, in_memory=args.in_memory,
                          drop_feature_group=drop, region=region)
    log(f"  events: train={len(train_ds)} eval={len(val_ds)} test={len(test_ds)}")
    log(f"  done in {time.time()-t:.1f}s")
    if args.subset_train:
        train_ds = Subset(train_ds, list(range(min(args.subset_train, len(train_ds)))))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, drop_last=True, persistent_workers=args.num_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers)

    edge_index, edge_dirs = build_grid_edges(connectivity=args.connectivity)
    edge_index = edge_index.to(device); edge_dirs = edge_dirs.to(device)

    kw = dict(
        hidden=args.hidden, heads=args.heads, n_layers=args.n_layers,
        ode_layers=args.ode_layers,
        t_end=args.t_end, n_eval_steps=args.n_eval_steps,
        monotone=args.monotone, uniform_edges=args.uniform_edges,
        adjoint=args.adjoint, solver=args.solver, base=args.hidden // 2,
        frobenius_weight=args.frobenius_weight,
    )
    model = make_model(args.model, edge_index, edge_dirs, norm_mean, norm_std, **kw)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"params: {n_params:,}")

    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    history = []
    best_val_aucpr = -1.0
    curves_path = out / "curves.csv"
    with open(curves_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_auc_pr", "val_auc_roc", "val_csi", "val_f1", "epoch_sec"])

    is_pignode = args.model.lower() == "pignode"
    use_frob = is_pignode and args.frobenius_weight > 0.0
    use_soft_mono = args.soft_mono_weight > 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        loss_breakdown = {"focal": [], "soft_mono": [], "frob": []}
        t0 = time.time()
        for i, (x, y) in enumerate(train_loader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            model_out = model(x)
            if isinstance(model_out, tuple):
                logits, dyn_norms = model_out
            else:
                logits, dyn_norms = model_out, None

            loss_focal = focal_bce_with_logits(
                logits, y, alpha=args.focal_alpha, gamma=args.focal_gamma
            )
            loss = loss_focal
            loss_breakdown["focal"].append(loss_focal.item())

            if use_soft_mono:
                # differentiable burn-irreversibility: push p>=1 on already-burning cells.
                # x[:,0] is PrevFireMask (kept binary; not standardized)
                lm = soft_monotonicity_penalty(logits, x[:, 0])
                loss = loss + args.soft_mono_weight * lm
                loss_breakdown["soft_mono"].append(lm.item())

            if use_frob and dyn_norms is not None and len(dyn_norms) > 0:
                lf = frobenius_dynamics_penalty(dyn_norms)
                loss = loss + args.frobenius_weight * lf
                loss_breakdown["frob"].append(lf.item())

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
            if args.max_steps and i + 1 >= args.max_steps:
                break
        sched.step()
        epoch_time = time.time() - t0
        m = evaluate(model, val_loader, device, max_batches=args.eval_batches)
        extras = ""
        if loss_breakdown["soft_mono"]:
            extras += f" mono {np.mean(loss_breakdown['soft_mono']):.4f}"
        if loss_breakdown["frob"]:
            extras += f" frob {np.mean(loss_breakdown['frob']):.4f}"
        log(f"epoch {epoch:3d}  loss {np.mean(losses):.4f}{extras}  "
            f"val AUC-PR {m['auc_pr']:.3f}  AUC-ROC {m['auc_roc']:.3f}  "
            f"CSI {m['csi']:.3f}  F1 {m['f1']:.3f}  ({epoch_time:.1f}s)")
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), **m, "epoch_sec": epoch_time})
        with open(curves_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, float(np.mean(losses)), m["auc_pr"], m["auc_roc"], m["csi"], m["f1"], epoch_time])
        if m["auc_pr"] > best_val_aucpr:
            best_val_aucpr = m["auc_pr"]
            torch.save({"state_dict": model.state_dict(), "args": vars(args), "metrics": m}, out / "best.pt")

    # final eval on best ckpt
    ckpt = torch.load(out / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    test_m = evaluate(model, test_loader, device)
    log(f"TEST  AUC-PR {test_m['auc_pr']:.3f}  AUC-ROC {test_m['auc_roc']:.3f}  "
        f"CSI {test_m['csi']:.3f}  F1 {test_m['f1']:.3f}")
    with open(out / "metrics.json", "w") as f:
        json.dump({"history": history, "test": test_m, "best_val": ckpt["metrics"], "args": vars(args)},
                  f, indent=2)


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=[
        "convae", "gcn", "sage", "gat",
        "gcn_edge", "sage_edge", "gat_edge",   # edge-aware variants
        "pignode",
    ])
    p.add_argument("--exp", default=None)
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--n-layers", type=int, default=3)
    p.add_argument("--ode-layers", type=int, default=2,
                   help="GAT layers stacked inside dh/dt")
    p.add_argument("--t-end", type=float, default=1.0)
    p.add_argument("--n-eval-steps", type=int, default=2)
    p.add_argument("--connectivity", type=int, default=8)
    p.add_argument("--monotone", action="store_true", default=True)
    p.add_argument("--no-monotone", dest="monotone", action="store_false")
    p.add_argument("--uniform-edges", action="store_true", default=False)
    p.add_argument("--adjoint", action="store_true", default=False)
    p.add_argument("--no-adjoint", dest="adjoint", action="store_false")
    p.add_argument("--solver", default="rk4", help="rk4 (MPS-safe) or dopri5 (CPU only)")
    p.add_argument("--focal-alpha", type=float, default=0.75)
    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument("--drop-feature-group", default=None,
                   choices=["topo", "weather", "fuel", "human"],
                   help="ablation: zero a feature group")
    p.add_argument("--region", default="all",
                   choices=["all", "high_elev", "low_elev"],
                   help="elevation-based domain split for cross-region tests")
    p.add_argument("--frobenius-weight", type=float, default=0.0,
                   help="lambda on ||dh/dt||^2")
    p.add_argument("--soft-mono-weight", type=float, default=0.0,
                   help="weight for the differentiable monotonicity penalty")
    p.add_argument("--max-steps", type=int, default=None, help="cap steps/epoch (debug)")
    p.add_argument("--eval-batches", type=int, default=None, help="cap eval batches (debug)")
    p.add_argument("--subset-train", type=int, default=None, help="use first N train events")
    p.add_argument("--in-memory", action="store_true", default=True)
    p.add_argument("--no-in-memory", dest="in_memory", action="store_false")
    args = p.parse_args()
    if args.exp is None:
        tags = [args.model]
        if not args.monotone:
            tags.append("no_mono")
        if args.uniform_edges:
            tags.append("uniform")
        if args.drop_feature_group:
            tags.append(f"drop_{args.drop_feature_group}")
        if args.connectivity != 8:
            tags.append(f"c{args.connectivity}")
        if args.region != "all":
            tags.append(args.region)
        if args.solver != "rk4":
            tags.append(args.solver)
        if args.frobenius_weight > 0:
            tags.append(f"frob{args.frobenius_weight:g}")
        if args.soft_mono_weight > 0:
            tags.append(f"sm{args.soft_mono_weight:g}")
        if args.n_layers != 3 and "pignode" not in args.model:
            tags.append(f"L{args.n_layers}")
        if "pignode" in args.model and args.ode_layers != 2:
            tags.append(f"ol{args.ode_layers}")
        args.exp = "_".join(tags)
    return args


if __name__ == "__main__":
    train(parse())
