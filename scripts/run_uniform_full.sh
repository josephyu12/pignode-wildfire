#!/usr/bin/env bash
# Best-variant on FULL data: uniform (no physics edges) + h96 + ode_layers=2 +
# monotonicity inference-only. The "uniform" variant won on 5K (0.220 > GAT 0.218).
# Pushing it to full data is the most likely path to a stronger headline number.
set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate

export PYTHONUNBUFFERED=1
LOG=experiments/_runlog.txt
echo "=== run_uniform_full started $(date) ===" >>"$LOG"

run () {
    local name="$1"; shift
    echo "[$(date +%H:%M:%S)] >>> $name" | tee -a "$LOG"
    python -m wildfire.train --exp "$name" "$@" 2>&1 | tee -a "$LOG"
    echo "[$(date +%H:%M:%S)] <<< $name done" | tee -a "$LOG"
}

run pignode_uniform_full --model pignode --uniform-edges \
    --hidden 96 --ode-layers 2 --n-eval-steps 1 --t-end 1.0 \
    --epochs 5 --batch-size 16 --lr 3e-4

echo "=== run_uniform_full done $(date) ===" >>"$LOG"
