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

# Proposal §2.1 feature groups, used by the feature-group drop ablation
# (proposal §6.3 #2). Dropping "fire" is *not* offered: the model can't
# predict next-day fire without seeing today's fire mask, so dropping it
# would be measuring noise rather than feature contribution.
FEATURE_GROUPS: dict[str, list[int]] = {
    "topo":    [1],            # elevation
    "weather": [2, 3, 4, 5, 6, 7],  # th, vs, tmmn, tmmx, sph, pr
    "fuel":    [8, 9, 10],     # pdsi, NDVI, erc
    "human":   [11],           # population
}

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


# ---------------------------------------------------------------------------
# Geographic / topographic domain-shift splits.
#
# NDWS does not ship lat/lon per event, so we use *elevation* (feature index 1)
# as a domain-shift proxy: events with above-median mean elevation are roughly
# the western/mountainous fires (steep terrain, slope-dominated spread); below-
# median events are roughly central/eastern lowland fires (flat terrain, wind-
# dominated spread). The split threshold is fixed on the *training* split's
# median to avoid test-set leakage; the same threshold is then applied to the
# eval and test splits, so an event keeps the same regional label across all
# uses.
# ---------------------------------------------------------------------------

REGION_FEATURE_IDX = 1   # "elevation"
REGION_NAMES = ("high_elev", "low_elev")


def compute_region_assignments(
    root: str | Path = "data/raw/ndws",
    cache_path: str | Path | None = None,
) -> dict:
    """Per-split index lists of events in each elevation region.

    Returns a dict with keys {split}_{region}, e.g. "train_high_elev",
    plus a "threshold" entry recording the elevation median used.
    Cached to data/processed/region_splits.npz.
    """
    cache_path = Path(cache_path or "data/processed/region_splits.npz")
    if cache_path.exists():
        d = np.load(cache_path)
        return {k: d[k] for k in d.files}

    out: dict = {}

    def _per_event_mean(split: str) -> np.ndarray:
        ds = xr.open_zarr(_zarr_path(split, root), consolidated=False)
        n = ds.sizes["time"]
        var = ds[FEATURE_ORDER[REGION_FEATURE_IDX]]
        # Streaming reduction so we never materialize all events at once.
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
    """Returns (x: (12,64,64) float32, y: (64,64) float32 in {0,1,-1}).

    -1 in the original Huot et al. release marks unlabeled/QA-fail pixels and is excluded
    from loss + metrics. The HF mirror appears to have replaced -1 with 0/1 only; we still
    handle -1 defensively.
    """

    def __init__(self, split: str = "train", root: str | Path = "data/raw/ndws",
                 normalize: bool = True, augment: bool = False, in_memory: bool = False,
                 drop_feature_group: str | None = None,
                 region: str | None = None):
        self.split = split
        self.ds = xr.open_zarr(_zarr_path(split, root), consolidated=False)
        self.full_n = int(self.ds.sizes["time"])
        # Region filter: events are kept iff their per-event mean elevation is
        # above (high_elev) or below (low_elev) the train-set median. This makes
        # PI-GNODE see only mountainous OR only flatland events for cross-region
        # generalization tests.
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
        # Build a per-feature multiplier for the proposal's feature-group ablation.
        # Zeroing post-normalization sets the dropped channel to its training-set
        # mean (since (mean - mean)/std = 0), which is the cleanest "no-information"
        # baseline the model could see.
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
            # _cached is already region-filtered, so idx is a local index.
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
        # Apply feature-group drop *after* normalization so dropped channels are
        # exactly zero (the standardized mean) rather than the raw mean.
        x = x * self.drop_mask
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
