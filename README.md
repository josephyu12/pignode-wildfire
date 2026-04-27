# pignode-wildfire

Physics-Informed Graph Neural ODE (PI-GNODE) for wildfire spread prediction.
CPSC 452 — Deep Learning, Spring 2026 final project.

Proposal: [`docs/proposal.pdf`](docs/proposal.pdf).

## Team

- Joseph Yu
- Edward Lai
- Steven Zhang
- Calum Zhang

## Plan — 7-day descope

The proposal is paper-scope (PI-GNODE + 6 baselines + 6 ablations + 2 datasets + multi-day rollout). With ~7 days we cut to a shippable core:

**Cut:** TS-SatFire dataset, multi-day rollout, 4 of 6 ablations.
**Keep:** Next Day Wildfire Spread (NDWS), all 6 baselines (LR / RF / ConvAE / GCN / SAGE / GAT), PI-GNODE with monotonicity, 2 ablations (monotonicity prior, edge encoding).

If we ship the core early, add scope back in this order: more ablations → multi-day rollout → TS-SatFire.

### Day-by-day starting suggestion

| Day | Joseph | Edward | Steven | Calum |
|-----|--------|--------|--------|-------|
| 1 | Repo + env + NDWS download | TFRecord → PyG `Data` pipeline | LR + RF baselines | ConvAE baseline (replicate Huot et al.) |
| 2 | GAT w/ edge features | GraphSAGE | GCN | Shared training loop + focal loss |
| 3 | PI-GNODE: GAT-derivative `f_θ` | torchdiffeq integration (dopri5 + adjoint) | Metrics: AUC-PR, CSI, F1@best-thresh | First end-to-end PI-GNODE run |
| 4 | Monotonicity constraint | Iterate on PI-GNODE | Run baselines at full scale | Hyperparam sweep |
| 5 | Ablation: monotonicity on/off | Ablation: uniform vs physics edge feats | Eval all models, build results table | Plots + qualitative viz |
| 6 | Final eval | Write-up | Write-up | Write-up |
| 7 | Buffer / polish / submit | | | |

Reassign as workload reveals itself.

## Setup

```bash
# Python 3.10+
python -m venv .venv && source .venv/bin/activate
pip install -e .[dev]
```

PyG note: on M1 Mac, the pure-Python `torch-geometric` package works without the C++ extensions for our use case. If you hit a missing-op error, install scatter/sparse from the PyG wheel index for your torch version.

## Data

NDWS (primary): https://www.kaggle.com/datasets/fantineh/next-day-wildfire-spread

Place TFRecord files under `data/raw/ndws/`. The pipeline in `src/wildfire/data/ndws.py` (TODO) parses to PyG `Data` objects.

## Hardware

**Default: M1 Max + MPS.** Sufficient for 4096-node graphs and 18.5K events. Don't burn a day on cluster setup unless we hit a wall.

**Fallback: Yale YCRC** (Grace / McCleary). If we move to YCRC, scripts go in `scripts/slurm/`. Code must work locally first.

## Repo structure

```
src/wildfire/
  data/         # TFRecord parsing, graph construction
  models/       # baselines, GNNs, PI-GNODE
  metrics.py    # AUC-PR, CSI, F1
  train.py
  eval.py
configs/        # YAML per experiment
scripts/        # SLURM scripts (when on YCRC)
notebooks/      # exploration + final viz
data/           # raw + processed (gitignored)
experiments/    # logged results (gitignored)
docs/           # proposal, write-up
```

## Metrics

Primary: **AUC-PR** (handles severe class imbalance — fire pixels are <2% of grid).
Secondary: AUC-ROC (Huot et al. comparability), CSI / Threat Score, F1 at validation-optimal threshold.

## Targets to beat (Huot et al.)

| Model | AUC | Precision | Recall |
|-------|-----|-----------|--------|
| Conv autoencoder | 0.284 | 0.336 | 0.431 |
| Random Forest | — | — | — |
| Logistic Regression | — | matched AE precision | lower |

PI-GNODE goal: beat ConvAE on AUC-PR and CSI.
