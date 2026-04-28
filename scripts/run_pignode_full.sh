#!/usr/bin/env bash
# v7: same h96 + 2-layer ODE + monotonicity-inference-only architecture as v6,
# but on the FULL 14,979-event train set. 5 epochs (compute budget).
# Then ablations on the 5K subset (matched to baseline conditions).
set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate

export PYTHONUNBUFFERED=1
LOG=experiments/_runlog.txt
echo "=== run_pignode_full v7 started $(date) ===" >>"$LOG"

run () {
    local name="$1"; shift
    echo "[$(date +%H:%M:%S)] >>> $name" | tee -a "$LOG"
    python -m wildfire.train --exp "$name" "$@" 2>&1 | tee -a "$LOG"
    echo "[$(date +%H:%M:%S)] <<< $name done" | tee -a "$LOG"
}

ARCH="--hidden 96 --ode-layers 2 --n-eval-steps 1 --t-end 1.0"

# Main run on FULL data
run pignode --model pignode --epochs 5 --batch-size 16 --lr 3e-4 $ARCH

# Ablations on 5K subset (same conditions as the GNN baselines)
SUB="--subset-train 5000 --eval-batches 30"
ABL="--model pignode --epochs 8 --batch-size 16 --lr 3e-4 $ARCH $SUB"
run pignode_no_mono $ABL --no-monotone
run pignode_uniform $ABL --uniform-edges

echo "=== run_pignode_full v7 done $(date) ===" >>"$LOG"
