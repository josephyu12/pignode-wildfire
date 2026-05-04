"""Eval metrics. -1 cells are unlabeled and dropped. Headline metric is AUC-PR."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)


def _flatten_valid(y_true: np.ndarray, y_score: np.ndarray):
    y_true = np.asarray(y_true).reshape(-1)
    y_score = np.asarray(y_score).reshape(-1)
    mask = y_true != -1
    return y_true[mask].astype(np.int8), y_score[mask].astype(np.float32)


def auc_pr(y_true, y_score) -> float:
    yt, ys = _flatten_valid(y_true, y_score)
    return float(average_precision_score(yt, ys))


def auc_roc(y_true, y_score) -> float:
    yt, ys = _flatten_valid(y_true, y_score)
    return float(roc_auc_score(yt, ys))


def csi(y_true, y_pred) -> float:
    """CSI = TP / (TP + FP + FN). Standard hazard-forecasting metric."""
    yt, yp = _flatten_valid(y_true, y_pred)
    yp = (yp >= 0.5).astype(np.int8)
    tp = int(((yt == 1) & (yp == 1)).sum())
    fp = int(((yt == 0) & (yp == 1)).sum())
    fn = int(((yt == 1) & (yp == 0)).sum())
    denom = tp + fp + fn
    return float(tp / denom) if denom > 0 else 0.0


def best_f1_threshold(y_true, y_score) -> tuple[float, float]:
    """Sweep thresholds -> (best_f1, threshold)."""
    yt, ys = _flatten_valid(y_true, y_score)
    p, r, t = precision_recall_curve(yt, ys)
    f1 = 2 * p * r / np.clip(p + r, 1e-12, None)
    idx = int(np.argmax(f1[:-1])) if len(t) > 0 else 0
    return float(f1[idx]), float(t[idx]) if len(t) > 0 else 0.5


def csi_at_threshold(y_true, y_score, threshold: float) -> float:
    yt, ys = _flatten_valid(y_true, y_score)
    yp = (ys >= threshold).astype(np.int8)
    return csi(yt, yp)


def all_metrics(y_true, y_score) -> dict:
    """Full metric panel."""
    yt, ys = _flatten_valid(y_true, y_score)
    f1, thr = best_f1_threshold(yt, ys)
    yp = (ys >= thr).astype(np.int8)
    return {
        "auc_pr": auc_pr(yt, ys),
        "auc_roc": auc_roc(yt, ys),
        "csi": csi_at_threshold(yt, ys, thr),
        "f1": f1,
        "threshold": thr,
        "f1_default": float(f1_score(yt, (ys >= 0.5).astype(np.int8), zero_division=0)),
        "n_pos": int(yt.sum()),
        "n_neg": int((yt == 0).sum()),
        "pos_frac": float(yt.mean()),
    }
