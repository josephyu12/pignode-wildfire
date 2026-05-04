"""NDWS (Huot et al. 2022) loader.

Reads the HF mirror's zarr files at data/raw/ndws/{train,eval,test}.zarr.
Each event: 64x64, 12 input features + a FireMask label at t+1.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset

FEATURE_ORDER = [
    "PrevFireMask",   # fire at t (binary)
    "elevation",
    "th",             # wind direction (deg)
    "vs",             # wind speed
    "tmmn", "tmmx",   # min/max temp
    "sph",            # specific humidity
    "pr",             # precipitation
    "pdsi",           # drought index
    "NDVI",
    "erc",            # energy release component
    "population",
]
TARGET = "FireMask"

# Feature groups for the "drop a group" ablation. We don't offer dropping
# fire itself — without today's fire mask there's nothing to predict from.
FEATURE_GROUPS: dict[str, list[int]] = {
    "topo":    [1],
    "weather": [2, 3, 4, 5, 6, 7],
    "fuel":    [8, 9, 10],
    "human":   [11],
}

_NORM_CACHE: dict | None = None


def _zarr_path(split: str, root: str | Path = "data/raw/ndws") -> Path:
    return Path(root) / f"{split}.zarr"


def compute_norm_stats(root: str | Path = "data/raw/ndws", cache_path: str | Path | None = None) -> dict:
    """Streaming mean/std on the train split, cached to processed/."""
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
        block = np.stack([ds[v].isel(time=sl).values for v in FEATURE_ORDER], axis=1)
        block = np.nan_to_num(block, nan=0.0)
        sums += block.sum(axis=(0, 2, 3))
        sumsq += (block.astype(np.float64) ** 2).sum(axis=(0, 2, 3))
        count += block.shape[0] * block.shape[2] * block.shape[3]
    mean = sums / count
    var = sumsq / count - mean ** 2
    std = np.sqrt(np.clip(var, 1e-8, None))
    # leave the binary fire mask alone
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


# NDWS doesn't ship lat/lon, so we use mean elevation as a stand-in for the
# western-mountains vs central-lowlands split. Threshold is the train median,
# applied to all splits to avoid leakage.

REGION_FEATURE_IDX = 1   # elevation
REGION_NAMES = ("high_elev", "low_elev")


def compute_region_assignments(
    root: str | Path = "data/raw/ndws",
    cache_path: str | Path | None = None,
) -> dict:
    """Per-split event indices for high/low elevation regions, cached to processed/."""
    cache_path = Path(cache_path or "data/processed/region_splits.npz")
    if cache_path.exists():
        d = np.load(cache_path)
        return {k: d[k] for k in d.files}

    out: dict = {}

    def _per_event_mean(split: str) -> np.ndarray:
        ds = xr.open_zarr(_zarr_path(split, root), consolidated=False)
        n = ds.sizes["time"]
        var = ds[FEATURE_ORDER[REGION_FEATURE_IDX]]
        means = np.empty(n, dtype=np.float32)
        chunk = 256
        for i in range(0, n, chunk):
            sl = slice(i, min(i + chunk, n))
            block = np.nan_to_num(var.isel(time=sl).values, nan=0.0)
            means[sl] = block.reshape(block.shape[0], -1).mean(axis=1)
        return means

    train_means = _per_event_mean("train")
    threshold = float(np.median(train_means))
    out["threshold"] = np.array(threshold, dtype=np.float32)

    for split in ("train", "eval", "test"):
        means = train_means if split == "train" else _per_event_mean(split)
        high = np.where(means >= threshold)[0].astype(np.int64)
        low = np.where(means < threshold)[0].astype(np.int64)
        out[f"{split}_high_elev"] = high
        out[f"{split}_low_elev"] = low

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, **out)
    return out


class NDWSDataset(Dataset):
    """(x: (12,64,64) float32, y: (64,64) float32 in {0,1,-1}).

    -1 marks unlabeled / QA-fail pixels in the original release. The HF
    mirror seems to use only {0,1} but we still handle -1 just in case.
    """

    def __init__(self, split: str = "train", root: str | Path = "data/raw/ndws",
                 normalize: bool = True, augment: bool = False, in_memory: bool = False,
                 drop_feature_group: str | None = None,
                 region: str | None = None):
        self.split = split
        self.ds = xr.open_zarr(_zarr_path(split, root), consolidated=False)
        self.full_n = int(self.ds.sizes["time"])
        if region is None or region == "all":
            self.region = None
            self._idx_in_zarr = np.arange(self.full_n, dtype=np.int64)
        elif region in REGION_NAMES:
            self.region = region
            assignments = compute_region_assignments(root)
            self._idx_in_zarr = assignments[f"{split}_{region}"]
        else:
            raise ValueError(f"unknown region {region!r}; valid: {REGION_NAMES + ('all', None)}")
        self.n = int(len(self._idx_in_zarr))
        self.augment = augment and split == "train"
        self.normalize = normalize
        self._in_memory = in_memory
        # zero a feature group post-normalization == replacing it with its
        # train-set mean ("no-information" baseline)
        self.drop_mask = np.ones((len(FEATURE_ORDER), 1, 1), dtype=np.float32)
        if drop_feature_group is not None:
            if drop_feature_group not in FEATURE_GROUPS:
                raise ValueError(
                    f"unknown feature group {drop_feature_group!r}; "
                    f"valid: {sorted(FEATURE_GROUPS)}"
                )
            for idx in FEATURE_GROUPS[drop_feature_group]:
                self.drop_mask[idx, 0, 0] = 0.0
        if in_memory:
            full = np.stack(
                [self.ds[v].values for v in FEATURE_ORDER + [TARGET]], axis=1
            ).astype(np.float32)
            self._cached = full[self._idx_in_zarr] if self.region else full
        if normalize:
            ns = get_norm()
            self.mean = ns["mean"].reshape(-1, 1, 1)
            self.std = ns["std"].reshape(-1, 1, 1)

    def __len__(self) -> int:
        return self.n

    def _load(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        if self._in_memory:
            # _cached is already region-filtered -> local idx
            stack = self._cached[idx]
            x = stack[: len(FEATURE_ORDER)]
            y = stack[-1]
        else:
            real = int(self._idx_in_zarr[idx])
            x = np.stack(
                [self.ds[v].isel(time=real).values for v in FEATURE_ORDER], axis=0
            ).astype(np.float32)
            y = self.ds[TARGET].isel(time=real).values.astype(np.float32)
        x = np.nan_to_num(x, nan=0.0)
        y = np.where(np.isnan(y), -1.0, y)
        return x, y

    def __getitem__(self, idx: int):
        x, y = self._load(idx)
        if self.normalize:
            x = (x - self.mean) / self.std
        # drop after normalization so a dropped channel == its standardized mean (0)
        x = x * self.drop_mask
        if self.augment:
            # rot/flip the image AND rotate wind direction with it
            k = np.random.randint(4)
            if k:
                x = np.rot90(x, k=k, axes=(1, 2)).copy()
                y = np.rot90(y, k=k, axes=(0, 1)).copy()
                x[2] = (x[2] - 90.0 * k) % 360.0
            if np.random.rand() < 0.5:
                x = x[:, :, ::-1].copy()
                y = y[:, ::-1].copy()
                x[2] = (-x[2]) % 360.0
        return torch.from_numpy(x), torch.from_numpy(y)


def loaders(batch_size: int = 32, num_workers: int = 0, in_memory: bool = False,
            drop_feature_group: str | None = None,
            region: str | None = None, **kw):
    from torch.utils.data import DataLoader

    train = NDWSDataset("train", augment=True, in_memory=in_memory,
                        drop_feature_group=drop_feature_group, region=region, **kw)
    val = NDWSDataset("eval", augment=False, in_memory=in_memory,
                      drop_feature_group=drop_feature_group, region=region, **kw)
    test = NDWSDataset("test", augment=False, in_memory=in_memory,
                       drop_feature_group=drop_feature_group, region=region, **kw)
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True),
        DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=num_workers),
    )
