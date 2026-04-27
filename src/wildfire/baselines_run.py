"""Train + eval LR / RF baselines on flattened pixels.

Writes experiments/{lr,rf}/metrics.json so figures.py picks them up.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader

from .data.ndws import NDWSDataset
from .metrics import all_metrics
from .models.baselines import collect_pixels, predict_full_split


def run(name: str, model, n_train_events: int = 1500):
    train_ds = NDWSDataset("train", augment=False)
    test_ds = NDWSDataset("test", augment=False)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)

    print(f"=== {name} ===")
    t = time.time()
    X, y = collect_pixels(train_loader, max_events=n_train_events,
                          max_pos=200_000, neg_per_pos=5, rng_seed=0)
    print(f"  collected {X.shape[0]:,} pixel samples ({y.mean():.3f} pos) in {time.time()-t:.1f}s")
    t = time.time()
    model.fit(X, y)
    print(f"  fit done in {time.time()-t:.1f}s")
    t = time.time()
    y_true, y_score = predict_full_split(test_loader, model)
    m = all_metrics(y_true, y_score)
    print(f"  TEST  AUC-PR {m['auc_pr']:.3f}  AUC-ROC {m['auc_roc']:.3f}  "
          f"CSI {m['csi']:.3f}  F1 {m['f1']:.3f}  ({time.time()-t:.1f}s)")
    out = Path("experiments") / name
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "metrics.json", "w") as f:
        json.dump({"history": [], "test": m, "best_val": m, "args": {"model": name}}, f, indent=2)


def main():
    run("lr", LogisticRegression(max_iter=2000, C=1.0, n_jobs=-1), n_train_events=2000)
    run("rf", RandomForestClassifier(n_estimators=200, max_depth=15, n_jobs=-1,
                                     class_weight="balanced", random_state=0),
        n_train_events=1500)


if __name__ == "__main__":
    main()
