"""TS-SatFire prediction-task loader (proposal §2.2, §4.4).

This loader consumes the *preprocessed* `.npy` files produced by the official
TS-SatFire pipeline (https://github.com/zhaoyutim/TS-SatFire), specifically
`dataset_gen_pred.py`. Field semantics, channel layout, normalization
constants, and label construction here all match that repo's
`satimg_dataset_processor/data_generator_pred_torch.py` so anyone reading the
TS-SatFire paper can verify them line by line.

Pipeline summary (run once, before training)
--------------------------------------------
1. Download the GeoTIFF dataset (~71 GB) from Kaggle:
       https://www.kaggle.com/datasets/z789456sx/ts-satfire
   Auth via `kaggle datasets download -d z789456sx/ts-satfire`.
2. Clone https://github.com/zhaoyutim/TS-SatFire and run, for each split:
       python dataset_gen_pred.py -mode train -ts 6 -it 3
       python dataset_gen_pred.py -mode val   -ts 6 -it 3
       python dataset_gen_pred.py -mode test  -ts 6 -it 3
   This produces `pred_<split>_img_seqtoseq_alll_6i_3.npy` (input) and a
   matching `..._label_seqtoseq...npy` (label) per split.
3. Move/symlink those .npy files under `data/raw/tssatfire/<split>/` and
   point `--root` at that directory.

Per-sample tensor layout (what __getitem__ returns)
---------------------------------------------------
    x: (T, F, H, W) float32       -- T-day input time series, F=27 channels
    y: (T, H, W)    float32       -- per-day "newly-burning" labels
    prev_fire: (H, W) float32    -- accumulated fire mask AT t=0 (channel 6)

The 27 input channels follow `data_generator_pred_torch.py`:
    0-5   : VIIRS surface-reflectance bands I1-I3 (day) + M11, I2, I1 (rolling-max)
    6     : VIIRS Day active fire (raw, NOT accumulated)
    7-8   : VIIRS Night thermal (max-tracked across the window)
    9-26  : 18 auxiliary FirePred channels
            (weather, topography, land-cover class @ idx 21,
             degree features @ idx 12, 18, 24)
The official normalization stats and "do-not-normalize" indices are copied
verbatim from that file.

Adapter to PI-GNODE's 12-channel NDWS schema
--------------------------------------------
PI-GNODE expects (B, 12, H, W) where channel 0 is PrevFireMask. The
`TSSatFirePIGNODEAdapter` projects TS-SatFire's 27 channels onto that
schema using the closest-corresponding fields:
    NDWS index   TS-SatFire field
    -----------  ----------------
    0 PrevFire   ch 6 (af) thresholded; first frame of the time-series
    1 elevation  one of the auxiliary topo channels (TODO: verify which)
    2 wind dir   ch 12 (degree feature, post-sin transform)
    ...
The exact mapping for indices 1-11 cannot be confirmed without reading the
auxiliary GeoTIFF metadata, which is not exposed in the published code. We
default to a *robust* 1-to-12 channel selector (`AUX_TO_NDWS_DEFAULT`) that
is overrideable from the CLI; expect to verify and possibly correct the
mapping after a small sanity-check run.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

# ---- Constants copied verbatim from TS-SatFire's data_generator_pred_torch.py ----
# https://github.com/zhaoyutim/TS-SatFire/blob/main/satimg_dataset_processor/
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
# Indices that must NOT be standardized: degree features get a sine transform
# instead, and channel 21 (land-cover class) is categorical.
TSSF_DEGREE_IDC = (12, 18, 24)
TSSF_LANDCOVER_IDX = 21
TSSF_DONT_NORMALIZE = set(TSSF_DEGREE_IDC) | {TSSF_LANDCOVER_IDX}

TSSF_N_CHANNELS = 27


def _split_paths(root: Path, split: str) -> tuple[Path, Path]:
    """Locate the (img, label) .npy pair for a split.

    The official pipeline names files like
        pred_<split>_img_seqtoseq_alll_<ts>i_<it>.npy
        pred_<split>_label_seqtoseq_alll_<ts>i_<it>.npy
    and we accept any matching glob in the split directory. If multiple
    pairs exist (e.g., different ts/it combinations) we pick the
    lexicographically first to be deterministic; users who care should be
    explicit and put exactly one pair per split dir.
    """
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
    """Reads the official TS-SatFire prediction-task .npy outputs.

    Memory-maps the .npy files (they can be many GB) and slices per-event.
    Returns the *raw 27-channel* time series; use TSSatFirePIGNODEAdapter
    to remap to PI-GNODE's 12-channel schema.
    """

    def __init__(self, split: str, root: str | Path = "data/raw/tssatfire",
                 normalize: bool = True, subset: int | None = None):
        img_path, lbl_path = _split_paths(Path(root), split)
        # mmap_mode='r' avoids loading 60+ GB into RAM.
        self.images = np.load(img_path, mmap_mode="r")
        self.labels = np.load(lbl_path, mmap_mode="r")
        # Official format after .transpose((0,2,1,3,4)): (N, C, T, H, W)
        if self.images.ndim != 5:
            raise ValueError(
                f"Expected (N,C,T,H,W) image array, got shape {self.images.shape}. "
                f"Did you run the official `dataset_gen_pred.py`? See module docstring."
            )
        # Optional uniform subsample. We use a deterministic stride so the
        # subset is well-spread across events rather than a single contiguous
        # block (the .npy is grouped by event ID).
        if subset is not None and subset < self.images.shape[0]:
            stride = max(1, self.images.shape[0] // subset)
            self._idx = np.arange(0, stride * subset, stride)[:subset]
        else:
            self._idx = np.arange(self.images.shape[0])
        self.n = len(self._idx)
        self.c, self.t, self.h, self.w = self.images.shape[1:]
        if self.c != TSSF_N_CHANNELS:
            raise ValueError(
                f"Expected {TSSF_N_CHANNELS} channels, got {self.c}. "
                f"TS-SatFire pred-task uses 8 spectral + 19 auxiliary = 27 channels."
            )
        self.normalize = normalize
        self.mean = TSSF_MEAN.reshape(-1, 1, 1, 1)
        self.std = TSSF_STD.reshape(-1, 1, 1, 1)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        real = int(self._idx[idx])
        x = np.array(self.images[real], dtype=np.float32)        # (C, T, H, W)
        y = np.array(self.labels[real], dtype=np.float32)        # (T, H, W)

        if self.normalize:
            for i in range(self.c):
                if i in TSSF_DONT_NORMALIZE:
                    continue
                x[i] = (x[i] - self.mean[i]) / self.std[i]
            # Sine-transform the degree features (matches TS-SatFire repo).
            for i in TSSF_DEGREE_IDC:
                x[i] = np.sin(np.deg2rad(x[i]))

        return torch.from_numpy(x), torch.from_numpy(y)


# ---------- Adapter from TS-SatFire's 27 channels to PI-GNODE's 12 ----------
# See module docstring for caveats. The default below picks one auxiliary
# channel per NDWS slot so PI-GNODE has *something* in every position; users
# should override after inspecting actual TS-SatFire raster metadata.
#
# NDWS slot      <-- TS-SatFire channel
AUX_TO_NDWS_DEFAULT: tuple[int, ...] = (
    6,    # 0 PrevFireMask   <-- ch 6 (active fire raster)
    19,   # 1 elevation      <-- one of the topo aux channels (verify)
    12,   # 2 wind dir (deg) <-- ch 12 (degree feature, sin-transformed in __getitem__)
    13,   # 3 wind speed     <-- ch 13 (close-by aux channel; verify)
    14,   # 4 tmmn           <-- ch 14 (verify)
    15,   # 5 tmmx           <-- ch 15 (verify)
    16,   # 6 sph            <-- ch 16 (verify)
    17,   # 7 pr             <-- ch 17 (verify)
    20,   # 8 pdsi           <-- ch 20 (verify)
    9,    # 9 NDVI           <-- ch 9  (long-window vegetation channel; verify)
    22,   # 10 erc           <-- ch 22 (verify)
    26,   # 11 population    <-- ch 26 (verify)
)


class TSSatFirePIGNODEAdapter(Dataset):
    """Wraps TSSatFireDataset to look like NDWSDataset to the rest of the code.

    For each TS-SatFire sample we return a 12-channel snapshot at day d,
    with channel 0 set to the *cumulative* burning mask through day d. This
    keeps the burn-irreversibility prior consistent (proposal eq. 4) when
    PI-GNODE rolls out across the time series.

    `multiday=True` returns the full (T, 12, H, W) sequence + (T, H, W) label
    sequence so the multi-day rollout (proposal §4.4) can be supervised at
    every day boundary; otherwise it returns a single random day in [1, T-1]
    for compatibility with the existing single-step training loop.
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
        """(C=27, H, W) -> (12, H, W) PI-GNODE schema with channel-0 overridden."""
        x12 = x_t[list(self.channel_map)].clone()
        x12[0] = prev_fire
        return x12

    def __getitem__(self, idx: int):
        x, y = self.base[idx]                    # (C, T, H, W), (T, H, W)
        T = x.shape[1]
        # Build cumulative fire mask: at day d, prev_fire is OR of labels up to d-1.
        cumfire = torch.zeros(T + 1, x.shape[2], x.shape[3], dtype=torch.float32)
        for d in range(T):
            cumfire[d + 1] = torch.maximum(cumfire[d], (y[d] > self.fire_threshold).float())

        if self.multiday:
            xs = torch.stack(
                [self._project(x[:, d], cumfire[d]) for d in range(T)], dim=0
            )
            return xs, y                         # (T, 12, H, W), (T, H, W)

        # Single-day: pick a uniformly random day with at least one burning seed
        # in `prev_fire` so the model's monotonicity prior is non-trivial.
        candidates = [d for d in range(T) if cumfire[d].sum() > 0]
        d = candidates[torch.randint(len(candidates), (1,)).item()] if candidates else 0
        return self._project(x[:, d], cumfire[d]), y[d]
