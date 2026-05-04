"""LR / RF / ConvAE baselines.

LR + RF run on flattened pixels (subsampled — there are ~76M pixels in train).
ConvAE is a small U-Net on (12, 64, 64).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- LR / RF (sklearn) ---

def collect_pixels(loader, max_events: int = 2000, max_pos: int | None = None,
                   neg_per_pos: int = 5, rng_seed: int = 0):
    """Flatten pixels into (12,) feature vectors, balanced via positive oversampling."""
    rng = np.random.default_rng(rng_seed)
    X_pos, X_neg = [], []
    n_done = 0
    for x, y in loader:
        x = x.numpy()  # (B, 12, 64, 64)
        y = y.numpy()
        B, _, H, W = x.shape
        for i in range(B):
            xi = x[i].reshape(12, -1).T  # (4096, 12)
            yi = y[i].reshape(-1)
            valid = yi != -1
            xi, yi = xi[valid], yi[valid]
            X_pos.append(xi[yi == 1])
            X_neg.append(xi[yi == 0])
            n_done += 1
            if n_done >= max_events:
                break
        if n_done >= max_events:
            break
    Xp = np.concatenate(X_pos, axis=0)
    Xn = np.concatenate(X_neg, axis=0)
    if max_pos is not None and Xp.shape[0] > max_pos:
        idx = rng.choice(Xp.shape[0], size=max_pos, replace=False)
        Xp = Xp[idx]
    n_neg = min(Xn.shape[0], Xp.shape[0] * neg_per_pos)
    idx = rng.choice(Xn.shape[0], size=n_neg, replace=False)
    Xn = Xn[idx]
    X = np.concatenate([Xp, Xn], axis=0)
    y = np.concatenate([np.ones(Xp.shape[0]), np.zeros(Xn.shape[0])], axis=0)
    perm = rng.permutation(X.shape[0])
    return X[perm].astype(np.float32), y[perm].astype(np.int8)


def predict_full_split(loader, model, max_events: int | None = None):
    """Per-pixel sklearn scores for the full split (or up to max_events)."""
    all_y, all_p = [], []
    n_done = 0
    for x, y in loader:
        x = x.numpy(); y = y.numpy()
        B = x.shape[0]
        flat = x.reshape(B, 12, -1).transpose(0, 2, 1).reshape(-1, 12)
        scores = model.predict_proba(flat)[:, 1].reshape(B, 64, 64)
        all_p.append(scores); all_y.append(y)
        n_done += B
        if max_events and n_done >= max_events:
            break
    return np.concatenate(all_y), np.concatenate(all_p)


# --- ConvAE ---

class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class ConvAE(nn.Module):
    """Tiny U-Net for the CNN baseline."""

    def __init__(self, in_ch: int = 12, base: int = 32):
        super().__init__()
        self.enc1 = ConvBlock(in_ch, base)
        self.enc2 = ConvBlock(base, base * 2)
        self.enc3 = ConvBlock(base * 2, base * 4)
        self.bot = ConvBlock(base * 4, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = ConvBlock(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = ConvBlock(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = ConvBlock(base * 2, base)
        self.head = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        b = self.bot(F.max_pool2d(e3, 2))
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1).squeeze(1)
