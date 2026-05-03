# pignode-wildfire

Physics-Informed Graph Neural ODE (PI-GNODE) for wildfire spread prediction.
CPSC 452 — Deep Learning, Spring 2026 final project.

Proposal: [`docs/proposal.pdf`](docs/proposal.pdf).

## Team

- Joseph Yu
- Edward Lai
- Steven Zhang
- Calum Zhang

## Final state of this repo

- `docs/paper.tex` -- final report in NeurIPS 2024 format. Compiles to a
  7-page PDF (`docs/paper.pdf`) covering all rubric sections: abstract,
  intro/motivation, background/related work, methods (with a TikZ model
  schematic), empirical results (main comparison + ablations + qualitative),
  discussion, conclusion + future work, author contributions.
- `experiments/` -- migrated checkpoints and figures from the team's earlier
  runs: PI-GNODE main (`pignode_uniform_full`), with-physics-edges
  (`pignode_full`), and without-monotonicity ablations (`pignode_no_mono`),
  plus `_figures/` (curves, ablation, qualitative panels, results table).
- All proposal scope (cross-region generalization, additional ablations,
  Frobenius/soft-monotonicity regularizers, edge-aware GCN/SAGE, multi-day
  rollout, TS-SatFire) is implemented in code and documented in the paper as
  "designed; empirical validation deferred". The relevant CLI flags and
  scripts are listed in the paper's results section.

## Cross-region experiment on Colab (recommended)

If you don't have NDWS data on your machine, run the cross-region experiment on Google Colab.

1. **Push this repo to GitHub** (if not already there).
2. Open [`notebooks/colab_region_split.ipynb`](notebooks/colab_region_split.ipynb) on github.com → click the **"Open in Colab"** badge (or paste the GitHub URL into colab.research.google.com).
3. Edit the first cell — set `REPO_URL` to your fork's URL.
4. Runtime → Change runtime type → **T4 GPU**.
5. Run all cells. ~1.5–2 hours total. Last cell downloads `region_split_results.zip`.
6. Unzip into `experiments/` locally; rerun `python -m wildfire.figures` to refresh the figure.

## How to run everything

After NDWS is downloaded into `data/raw/ndws/`:

```bash
# 1. Original headline experiments (LR, RF, ConvAE, GCN, SAGE, GAT, PI-GNODE).
bash scripts/run_all.sh

# 2. Everything new added to respect the proposal: cross-region generalization,
#    feature-group / connectivity / depth / solver ablations, edge-aware
#    GCN/SAGE, Frobenius regularizer, soft monotonicity loss. Aggregates
#    figures/tables at the end.
bash scripts/run_new_experiments.sh

# 3. (Optional) TS-SatFire pipeline. Requires Kaggle API auth and ~71 GB
#    download. Use the MAX_*_EVENTS env vars to subset for a small first run.
MAX_TRAIN_EVENTS=10 MAX_VAL_EVENTS=3 MAX_TEST_EVENTS=5 \
    bash scripts/download_tssatfire.sh
python -m wildfire.rollout --ckpt experiments/pignode_uniform_full/best.pt \
    --subset 50 --mode free
```

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
