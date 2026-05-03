#!/usr/bin/env bash
# Run every experiment that was added to "respect the proposal" but is not
# yet covered by the existing run_all.sh. Order is by ROI (cheapest /
# highest-value first), so even partial completion gives you something
# useful to put in the paper.
#
#   1. Cross-region generalization (replaces TS-SatFire as second-dataset story)
#      ~2x the cost of one PI-GNODE main run.
#   2. Proposal §6.3 ablations (feature drop, connectivity, depth, solver)
#      plus Frobenius dynamics regularizer + soft monotonicity loss.
#   3. Edge-aware GCN/SAGE baselines (tests proposal §4.2's edge-features
#      claim on non-attention GNNs).
#
# The TS-SatFire pipeline is intentionally NOT in this script -- run
# scripts/download_tssatfire.sh separately if you decide to commit to it.
# NOT using `set -e` -- partial completion is much better than nothing.
cd "$(dirname "$0")/.."

if [ ! -f .venv/bin/activate ]; then
    echo "Set up the venv first:  python -m venv .venv && pip install -e .[dev]"
    exit 1
fi

LOG=experiments/_runlog.txt
mkdir -p experiments
echo "=== run_new_experiments started $(date) ===" >>"$LOG"

echo "==== 1. Cross-region generalization (~2 PI-GNODE main runs) ====" | tee -a "$LOG"
bash scripts/run_region_split.sh || echo "region split phase had failures" | tee -a "$LOG"

echo "==== 2. Proposal §6.3 ablations + Frobenius + soft-mono ====" | tee -a "$LOG"
bash scripts/ablations.sh || echo "ablation phase had failures" | tee -a "$LOG"

echo "==== 3. Aggregate figures and tables ====" | tee -a "$LOG"
source .venv/bin/activate
python -m wildfire.figures 2>&1 | tee -a "$LOG" || \
    echo "figure aggregation failed (non-fatal)" | tee -a "$LOG"

echo "=== run_new_experiments done $(date) ===" | tee -a "$LOG"

echo "Done. Outputs:"
echo "  experiments/_figures/results_table.{csv,tex}"
echo "  experiments/_figures/curves.png"
echo "  experiments/_figures/ablation.png"
echo "  experiments/_figures/region_split.png"
echo "  experiments/_figures/qualitative_pignode_uniform_full.png"
