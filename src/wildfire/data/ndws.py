"""Next Day Wildfire Spread loader.

Source: HuggingFace mirror TheRootOf3/next-day-wildfire-spread (zarr-zipped Huot et al. 2022).
Splits live at data/raw/ndws/{train,eval,test}.zarr after unzipping.

Each event is 64x64 with 12 input features and a FireMask label at t+1.
Feature ordering follows Huot et al.:
    [PrevFireMask, elevation, th, vs, tmmn, tmmx, sph, pr, pdsi, NDVI, erc, population]
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset

FEATURE_ORDER = [
    "PrevFireMask",   # fire state at t
    "elevation",      # topography
    "th",             # wind direction (deg)
    "vs",             # wind speed
    "tmmn",           # min temp
    "tmmx",           # max temp
    "sph",            # specific humidity
    "pr",             # precipitation
    "pdsi",           # drought index
    "NDVI",           # vegetation
    "erc",            # energy release component
    "population",     # population density
]
TARGET = "FireMask"

# Per-feature normalization stats (mean, std) computed once on train split via compute_norm_stats().
# Updated lazily on first call; cached at data/processed/norm_stats.npz.
_NORM_CACHE: dict | None = None


def _zarr_path(split: str, root: str | Path = "data/raw/ndws") -> Path:
    return Path(root) / f"{split}.zarr"


def compute_norm_stats(root: str | Path = "data/raw/ndws", cache_path: str | Path | None = None) -> dict:
    """Streaming mean/std over the train split. Saves to processed/ for reuse."""
    cache_path = Path(cache_path or "data/processed/norm_stats.npz")
    if cache_path.exists():
        d = np.load(cache_path)
        return {"mean": d["mean"], "std": d["std"]}

    ds = xr.open_zarr(_zarr_path("train", root), consolidated=False)
    n = ds.sizes["time"]
    sums = np.zeros(len(FEATURE_ORDER), dtype=np.float64)
    sumsq = np.zeros(len(FEATURE_ORDER), dtype=np.float64)
    count = 0
    chunk = 512
    for i in range(0, n, chunk):
        sl = slice(i, min(i + chunk, n))
        block = np.stack([ds[v].isel(time=sl).values for v in FEATURE_ORDER], axis=1)  # (B,12,64,64)
        # PrevFireMask is binary; we still standardize the rest. Mask out NaNs.
        block = np.nan_to_num(block, nan=0.0)
        sums += block.sum(axis=(0, 2, 3))
        sumsq += (block.astype(np.float64) ** 2).sum(axis=(0, 2, 3))
        count += block.shape[0] * block.shape[2] * block.shape[3]
    mean = sums / count
    var = sumsq / count - mean ** 2
    std = np.sqrt(np.clip(var, 1e-8, None))
    # Don't normalize binary fire mask
    mean[0] = 0.0
    std[0] = 1.0
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, mean=mean.astype(np.float32), std=std.astype(np.float32))
    return {"mean": mean.astype(np.float32), "std": std.astype(np.float32)}


def get_norm() -> dict:
    global _NORM_CACHE
    if _NORM_CACHE is None:
        _NORM_CACHE = compute_norm_stats()
    return _NORM_CACHE


class NDWSDataset(Dataset):
    """Returns (x: (12,64,64) float32, y: (64,64) float32 in {0,1,-1}).

    -1 in the original Huot et al. release marks unlabeled/QA-fail pixels and is excluded
    from loss + metrics. The HF mirror appears to have replaced -1 with 0/1 only; we still
    handle -1 defensively.
    """

    def __init__(self, split: str = "train", root: str | Path = "data/raw/ndws",
                 normalize: bool = True, augment: bool = False, in_memory: bool = False):
        self.split = split
        self.ds = xr.open_zarr(_zarr_path(split, root), consolidated=False)
        self.n = int(self.ds.sizes["time"])
        self.augment = augment and split == "train"
        self.normalize = normalize
        self._in_memory = in_memory
        if in_memory:
            self._cached = np.stack(
                [self.ds[v].values for v in FEATURE_ORDER + [TARGET]], axis=1
            ).astype(np.float32)
        if normalize:
            ns = get_norm()
            self.mean = ns["mean"].reshape(-1, 1, 1)
            self.std = ns["std"].reshape(-1, 1, 1)

    def __len__(self) -> int:
        return self.n

    def _load(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        if self._in_memory:
            stack = self._cached[idx]
            x = stack[: len(FEATURE_ORDER)]
            y = stack[-1]
        else:
            x = np.stack(
                [self.ds[v].isel(time=idx).values for v in FEATURE_ORDER], axis=0
            ).astype(np.float32)
            y = self.ds[TARGET].isel(time=idx).values.astype(np.float32)
        x = np.nan_to_num(x, nan=0.0)
        y = np.where(np.isnan(y), -1.0, y)
        return x, y

    def __getitem__(self, idx: int):
        x, y = self._load(idx)
        if self.normalize:
            x = (x - self.mean) / self.std
        if self.augment:
            # Random flips + 90deg rotations preserve fire-spread physics if we also rotate
            # wind direction (feature index 2 = "th", degrees from north). Cheap +acc gain.
            k = np.random.randint(4)
            if k:
                x = np.rot90(x, k=k, axes=(1, 2)).copy()
                y = np.rot90(y, k=k, axes=(0, 1)).copy()
                # Rotate wind direction (degrees) by k*90 in the *image* plane.
                x[2] = (x[2] - 90.0 * k) % 360.0
            if np.random.rand() < 0.5:
                x = x[:, :, ::-1].copy()
                y = y[:, ::-1].copy()
                x[2] = (-x[2]) % 360.0
        return torch.from_numpy(x), torch.from_numpy(y)


def loaders(batch_size: int = 32, num_workers: int = 0, in_memory: bool = False, **kw):
    from torch.utils.data import DataLoader

    train = NDWSDataset("train", augment=True, in_memory=in_memory, **kw)
    val = NDWSDataset("eval", augment=False, in_memory=in_memory, **kw)
    test = NDWSDataset("test", augment=False, in_memory=in_memory, **kw)
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True),
        DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=num_workers),
    )
