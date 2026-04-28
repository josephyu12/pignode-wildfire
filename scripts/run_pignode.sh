#!/usr/bin/env bash
# Re-run PI-GNODE main + ablations with v2 config:
# - t_end=2.0 + 2-step RK4 (8 NFE) for deeper effective message passing
# - LR 1e-3 (3x higher; v1 was undertrained)
# - 8 epochs (was 6)
# - LayerNorm inside ODE removed (was clamping derivative magnitudes)
set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate

export PYTHONUNBUFFERED=1
LOG=experiments/_runlog.txt
echo "=== run_pignode v2 started $(date) ===" >>"$LOG"

run () {
    local name="$1"; shift
    echo "[$(date +%H:%M:%S)] >>> $name" | tee -a "$LOG"
    python -m wildfire.train --exp "$name" "$@" 2>&1 | tee -a "$LOG"
    echo "[$(date +%H:%M:%S)] <<< $name done" | tee -a "$LOG"
}

SUB="--subset-train 5000 --eval-batches 30"
# v6: keep v5 fixes (monotonicity inference-only, stacked 2-layer ODE) and bump
# hidden 64 -> 96 to close capacity gap with GAT (52K -> 99K params). 8 epochs.
PIG="--model pignode --epochs 8 --batch-size 16 --hidden 96 --ode-layers 2 --n-eval-steps 1 --t-end 1.0 --lr 3e-4 $SUB"

run pignode         $PIG
run pignode_no_mono $PIG --no-monotone
run pignode_uniform $PIG --uniform-edges

echo "=== run_pignode v2 done $(date) ===" >>"$LOG"
