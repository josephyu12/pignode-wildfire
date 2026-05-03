#!/usr/bin/env bash
# Cross-region generalization experiment (replaces TS-SatFire as the
# proposal's "second dataset" claim — same NDWS underlying data, but split
# by per-event mean elevation as a proxy for western mountainous vs
# eastern lowland fire regimes).
#
# Trains the headline PI-GNODE config twice (once per region), then runs
# every (trained_on, evaluated_on) combination of the test split. The 2x2
# matrix lets us read off both in-domain performance and the cross-region
# generalization gap.
#
# Output: experiments/pignode_<region>/{best.pt, metrics.json,
#         eval_test_<region>.json}
# Resilient by design: a single failed step (OOM, NaN, missing data) should
# not nuke the rest of the experiment. Each run logs success/fail and we
# aggregate at the end so partial completion still produces a useful figure.
cd "$(dirname "$0")/.."
source .venv/bin/activate
export PYTHONUNBUFFERED=1
LOG=experiments/_runlog.txt
mkdir -p experiments

if [ ! -d data/raw/ndws/train.zarr ]; then
    echo "ERROR: data/raw/ndws/train.zarr not found." | tee -a "$LOG"
    echo "Download NDWS first, then re-run.  See README." | tee -a "$LOG"
    exit 1
fi

echo "=== run_region_split started $(date) ===" >>"$LOG"

run () {
    local name="$1"; shift
    echo "[$(date +%H:%M:%S)] >>> $name" | tee -a "$LOG"
    if python -m wildfire.train --exp "$name" "$@" 2>&1 | tee -a "$LOG"; then
        echo "[$(date +%H:%M:%S)] <<< $name done" | tee -a "$LOG"
    else
        echo "[$(date +%H:%M:%S)] !!! $name FAILED (continuing)" | tee -a "$LOG"
    fi
}

ev () {
    local ckpt="$1"; local region="$2"
    echo "[$(date +%H:%M:%S)] >>> eval $ckpt on $region" | tee -a "$LOG"
    if [ ! -f "$ckpt" ]; then
        echo "  $ckpt missing (training likely failed); skipping" | tee -a "$LOG"
        return
    fi
    python -m wildfire.eval_region --ckpt "$ckpt" --region "$region" --split test \
        2>&1 | tee -a "$LOG" || \
        echo "  eval failed for $ckpt on $region (continuing)" | tee -a "$LOG"
}

# Headline PI-GNODE config (mirrors run_final.sh):
PIG="--model pignode --uniform-edges --hidden 128 --ode-layers 2 \
     --n-eval-steps 1 --t-end 1.0 --epochs 8 --batch-size 16 --lr 3e-4 \
     --eval-batches 30"

# 1. Train per-region. We don't subset further inside each region because
#    the region halves the dataset already (~7.5K train events per region).
run pignode_high_elev $PIG --region high_elev
run pignode_low_elev  $PIG --region low_elev

# 2. Cross-evaluation matrix (4 cells). The diagonal is in-domain test
#    metrics; the off-diagonal is the cross-region generalization claim.
ev experiments/pignode_high_elev/best.pt high_elev
ev experiments/pignode_high_elev/best.pt low_elev
ev experiments/pignode_low_elev/best.pt  high_elev
ev experiments/pignode_low_elev/best.pt  low_elev

echo "=== run_region_split done $(date) ===" | tee -a "$LOG"
# Aggregate region-split heatmap from whatever cells of the 2x2 finished.
python -m wildfire.figures 2>&1 | tee -a "$LOG" || \
    echo "figure aggregation failed (non-fatal)" | tee -a "$LOG"
