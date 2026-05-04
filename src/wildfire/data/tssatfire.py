"""TS-SatFire loader.

Consumes the .npy outputs from the official TS-SatFire `dataset_gen_pred.py`
(https://github.com/zhaoyutim/TS-SatFire). Channel layout, normalization
constants, and "don't normalize" indices below are copied verbatim from that
repo's `data_generator_pred_torch.py`.

One-time setup:
    1. Download from Kaggle: `kaggle datasets download -d z789456sx/ts-satfire`
    2. Clone TS-SatFire and run dataset_gen_pred.py per split (-ts 6 -it 3).
    3. Drop the .npy files under data/raw/tssatfire/{split}/.

__getitem__ returns x: (T, 27, H, W) and y: (T, H, W).

The 27 input channels:
    0-5   : VIIRS surface-reflectance (I1-I3 day + M11/I2/I1 rolling-max)
    6     : VIIRS Day active fire (raw, not accumulated)
    7-8   : VIIRS Night thermal (max across window)
    9-26  : 18 aux FirePred channels (weather/topo/land-cover at 21,
            degree features at 12, 18, 24)

TSSatFirePIGNODEAdapter remaps these to PI-GNODE's 12-channel NDWS schema.
The aux index mapping is best-guess (no GeoTIFF metadata in the upstream
repo) — override AUX_TO_NDWS_DEFAULT if the sanity check looks off.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

# Stats copied straight from TS-SatFire's data_generator_pred_torch.py.
TSSF_MEAN = np.array([
    18.224253, 26.95519, 20.09066, 318.25967, 308.78717, 14.165086,
    291.29214, 288.97382, 5110.5547, 2556.2627, 0.3907487, 3.4994626,
    216.23518, 276.5463, 291.8275, 70.32086, 0.0054306216, 10.120554,
    175.33012, 1290.8367, -1.5219007, 7.3989105, 7.584937, 1.4395763,
    3.306973, 19.259102, 0.0057929577,
], dtype=np.float32)
TSSF_STD = np.array([
    15.438321, 14.408274, 10.552524, 13.1312475, 12.155249, 9.652911,
    12.435288, 8.750125, 2400.766, 1206.8983, 2.37979, 1.6343528,
    85.730644, 47.332256, 50.045837, 22.48386, 0.0021515382, 8.429097,
    104.73222, 823.01483, 1.9954495, 4.1257873, 26.547232, 1.2017097,
    48.207355, 5.4114914, 0.0017134654,
], dtype=np.float32)
# degree features get a sin() transform; land-cover (21) is categorical
TSSF_DEGREE_IDC = (12, 18, 24)
TSSF_LANDCOVER_IDX = 21
TSSF_DONT_NORMALIZE = set(TSSF_DEGREE_IDC) | {TSSF_LANDCOVER_IDX}

TSSF_N_CHANNELS = 27


def _split_paths(root: Path, split: str) -> tuple[Path, Path]:
    """Find the (img, label) .npy pair for a split. Picks lexicographically
    first if there are multiple ts/it combinations in the dir."""
    sd = root / split
    if not sd.is_dir():
        raise FileNotFoundError(
            f"TS-SatFire split directory not found: {sd}\n"
            f"Run scripts/download_tssatfire.sh first, or follow the "
            f"docstring in src/wildfire/data/tssatfire.py."
        )
    imgs = sorted(sd.glob("*_img_seqtoseq_*.npy"))
    labels = sorted(sd.glob("*_label_seqtoseq_*.npy"))
    if not imgs or not labels:
        raise FileNotFoundError(
            f"No preprocessed .npy pair found in {sd}. "
            f"Expected files matching *_img_seqtoseq_*.npy + *_label_seqtoseq_*.npy."
        )
    return imgs[0], labels[0]


class TSSatFireDataset(Dataset):
    """Raw 27-channel TS-SatFire time series. Use the adapter for PI-GNODE."""

    def __init__(self, split: str, root: str | Path = "data/raw/tssatfire",
                 normalize: bool = True, subset: int | None = None):
        img_path, lbl_path = _split_paths(Path(root), split)
        # mmap so we don't pull 60+ GB into RAM
        self.images = np.load(img_path, mmap_mode="r")
        self.labels = np.load(lbl_path, mmap_mode="r")
        # expect (N, C, T, H, W) after the upstream transpose
        if self.images.ndim != 5:
            raise ValueError(
                f"expected (N,C,T,H,W), got {self.images.shape}. "
                f"Did you run dataset_gen_pred.py? See the module docstring."
            )
        # deterministic-stride subsample so we spread across event IDs
        if subset is not None and subset < self.images.shape[0]:
            stride = max(1, self.images.shape[0] // subset)
            self._idx = np.arange(0, stride * subset, stride)[:subset]
        else:
            self._idx = np.arange(self.images.shape[0])
        self.n = len(self._idx)
        self.c, self.t, self.h, self.w = self.images.shape[1:]
        if self.c != TSSF_N_CHANNELS:
            raise ValueError(
                f"expected {TSSF_N_CHANNELS} channels (8 spectral + 19 aux), got {self.c}"
            )
        self.normalize = normalize
        self.mean = TSSF_MEAN.reshape(-1, 1, 1, 1)
        self.std = TSSF_STD.reshape(-1, 1, 1, 1)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        real = int(self._idx[idx])
        x = np.array(self.images[real], dtype=np.float32)    # (C, T, H, W)
        y = np.array(self.labels[real], dtype=np.float32)    # (T, H, W)

        if self.normalize:
            for i in range(self.c):
                if i in TSSF_DONT_NORMALIZE:
                    continue
                x[i] = (x[i] - self.mean[i]) / self.std[i]
            # sin-transform degree features (matches TS-SatFire upstream)
            for i in TSSF_DEGREE_IDC:
                x[i] = np.sin(np.deg2rad(x[i]))

        return torch.from_numpy(x), torch.from_numpy(y)


# Best-guess mapping from TS-SatFire's 27 channels onto NDWS's 12 slots.
# The "verify" notes are real — we don't have GeoTIFF metadata for the aux
# rasters, so these picks may be wrong. Override if a sanity check fails.
AUX_TO_NDWS_DEFAULT: tuple[int, ...] = (
    6,    # 0 PrevFireMask  <- active fire
    19,   # 1 elevation     <- topo aux (verify)
    12,   # 2 wind dir      <- degree feature
    13,   # 3 wind speed    (verify)
    14,   # 4 tmmn          (verify)
    15,   # 5 tmmx          (verify)
    16,   # 6 sph           (verify)
    17,   # 7 pr            (verify)
    20,   # 8 pdsi          (verify)
    9,    # 9 NDVI          (verify)
    22,   # 10 erc          (verify)
    26,   # 11 population   (verify)
)


class TSSatFirePIGNODEAdapter(Dataset):
    """Make TSSatFireDataset look like NDWSDataset.

    Channel 0 of every snapshot is set to the *cumulative* burning mask
    through day d, so monotonicity stays consistent across the rollout.

    multiday=True -> returns full (T, 12, H, W) + (T, H, W).
    multiday=False -> picks a random day with at least one burning seed.
    """

    def __init__(self, base: TSSatFireDataset, channel_map=AUX_TO_NDWS_DEFAULT,
                 multiday: bool = False, fire_threshold: float = 0.5):
        if len(channel_map) != 12:
            raise ValueError("channel_map must select exactly 12 NDWS slots.")
        self.base = base
        self.channel_map = tuple(channel_map)
        self.multiday = multiday
        self.fire_threshold = fire_threshold

    def __len__(self) -> int:
        return len(self.base)

    def _project(self, x_t: torch.Tensor, prev_fire: torch.Tensor) -> torch.Tensor:
        x12 = x_t[list(self.channel_map)].clone()
        x12[0] = prev_fire
        return x12

    def __getitem__(self, idx: int):
        x, y = self.base[idx]
        T = x.shape[1]
        # cumfire[d] = OR of labels up to d-1
        cumfire = torch.zeros(T + 1, x.shape[2], x.shape[3], dtype=torch.float32)
        for d in range(T):
            cumfire[d + 1] = torch.maximum(cumfire[d], (y[d] > self.fire_threshold).float())

        if self.multiday:
            xs = torch.stack(
                [self._project(x[:, d], cumfire[d]) for d in range(T)], dim=0
            )
            return xs, y

        # pick a random day that already has a burning seed
        candidates = [d for d in range(T) if cumfire[d].sum() > 0]
        d = candidates[torch.randint(len(candidates), (1,)).item()] if candidates else 0
        return self._project(x[:, d], cumfire[d]), y[d]
