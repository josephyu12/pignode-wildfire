# pignode-wildfire

Physics-Informed Graph Neural ODE (PI-GNODE) for wildfire spread prediction.
CPSC 452 — Deep Learning, Spring 2026 final project, Yale University.

## Authors

Joseph Yu, Calum Zhang, Edward Lai, Steven Zhang.

## Paper

[`docs/paper.pdf`](docs/paper.pdf) (NeurIPS 2024 format, 5-page main body
+ appendix). Source: [`docs/paper.tex`](docs/paper.tex).

## Headline results

On the Next Day Wildfire Spread (NDWS) test set:

| Model | AUC-PR | AUC-ROC | CSI | F1 |
|---|---|---|---|---|
| Logistic Regression | 0.148 | 0.798 | 0.185 | 0.313 |
| Random Forest | 0.150 | 0.813 | 0.184 | 0.311 |
| Conv. Autoencoder | **0.357** | **0.944** | **0.262** | **0.415** |
| GCN | 0.208 | 0.837 | 0.204 | 0.338 |
| GraphSAGE | 0.229 | 0.857 | 0.212 | 0.349 |
| GAT | 0.218 | 0.854 | 0.211 | 0.349 |
| **PI-GNODE (ours)** | **0.221** | **0.896** | 0.206 | 0.341 |

**Cross-region generalization** (NDWS partitioned by per-event mean
elevation, then cross-evaluated): all four (train, test) combinations
achieve ~11× lift over random AUC-PR baseline, indicating the learned
spread dynamics transfer cleanly between mountainous and lowland fire
regimes. See paper §4.4 for details.

## Repository structure

```
docs/
  paper.tex, paper.pdf      # final report
  figures/schematic.tex     # TikZ model schematic
  neurips_2024.sty          # style file
src/wildfire/
  data/ndws.py              # NDWS zarr loader, region-split logic
  models/pignode.py         # PI-GNODE model
  models/gnns.py            # GCN / SAGE / GAT baselines
  models/baselines.py       # ConvAE / LR / RF baselines
  graph.py                  # grid -> graph construction, edge features
  losses.py                 # focal BCE loss
  metrics.py                # AUC-PR, CSI, F1
  train.py                  # training entry point
  eval_region.py            # cross-region evaluation
  figures.py                # aggregate figures + tables from JSON metrics
scripts/
  run_all.sh                # train all baselines + PI-GNODE
  run_region_split.sh       # cross-region experiment (4 runs + cross-eval)
notebooks/
  colab_region_split.ipynb  # one-click reproduction on Google Colab T4
experiments/
  pignode_uniform_full/     # PI-GNODE main checkpoint + metrics
  pignode_full/             # +physics-edges ablation
  pignode_no_mono/          # -monotonicity ablation
  pignode_high_elev/        # cross-region: trained on mountainous fires
  pignode_low_elev/         # cross-region: trained on lowland fires
  _figures/                 # PNGs + CSV tables used in the paper
```

## Reproducing the experiments

### Easiest path: Colab (no local setup needed)

For the cross-region experiment specifically, [`notebooks/colab_region_split.ipynb`](notebooks/colab_region_split.ipynb) does the full pipeline (data download, training, cross-evaluation, heatmap) in one click. Free-tier T4 GPU, ~2 hours wall-clock.

1. Open the notebook on github.com → click "Open in Colab".
2. Runtime → Change runtime type → T4 GPU.
3. Run all cells.

### Local path (requires NDWS data)

```bash
# 1. Set up environment
python -m venv .venv && source .venv/bin/activate
pip install -e .[dev]

# 2. Download NDWS into data/raw/ndws/. The HuggingFace mirror is
#    TheRootOf3/next-day-wildfire-spread; see notebooks/colab_region_split.ipynb
#    for the exact download + unzip steps (HuggingFace zarr.zip layout).

# 3. Run baselines + PI-GNODE
bash scripts/run_all.sh

# 4. Cross-region experiment
bash scripts/run_region_split.sh

# 5. Aggregate figures
python -m wildfire.figures
```

### Compiling the paper

```bash
cd docs && pdflatex paper.tex && pdflatex paper.tex
```

The NeurIPS style file is committed in `docs/neurips_2024.sty`; the
TikZ schematic is in `docs/figures/schematic.tex` and is included via
`\input{figures/schematic}`.

## Hardware notes

The reported PI-GNODE main run used `hidden=96`, `batch_size=16`, 5 epochs,
~3 hrs on Apple M1 Max with MPS backend. The cross-region runs used
`hidden=96`, `batch_size=8`, 8 epochs, ~1 hr each on a Colab T4 GPU. T4
needs the smaller batch to avoid OOM at hidden 128.
