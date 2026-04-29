#!/usr/bin/env bash
# Final PI-GNODE main run: h128 uniform (no physics edges) on 5K subset, 8 epochs.
# Sweep showed h128 +0.023 over h96 at 2K/3ep -- extrapolating to 5K/8ep, expect
# AUC-PR ~0.24 vs h96's 0.220.
set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate

export PYTHONUNBUFFERED=1
LOG=experiments/_runlog.txt
echo "=== run_final started $(date) ===" >>"$LOG"

run () {
    local name="$1"; shift
    echo "[$(date +%H:%M:%S)] >>> $name" | tee -a "$LOG"
    python -m wildfire.train --exp "$name" "$@" 2>&1 | tee -a "$LOG"
    echo "[$(date +%H:%M:%S)] <<< $name done" | tee -a "$LOG"
}

# Final main: h128 + 2-layer ODE + uniform edges + monotonicity inference-only
run pignode --model pignode --uniform-edges --hidden 128 --ode-layers 2 \
    --n-eval-steps 1 --t-end 1.0 \
    --epochs 8 --batch-size 16 --lr 3e-4 \
    --subset-train 5000 --eval-batches 30

echo "=== run_final done $(date) ===" >>"$LOG"
