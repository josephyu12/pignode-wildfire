#!/usr/bin/env bash
# Fast hyperparameter sweep on a 2K subset with 3 epochs each.
# Goal: find the variant that beats current best (uniform 5K = 0.220 test AUC-PR)
# without 2.5-hour runs per experiment.
set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate

export PYTHONUNBUFFERED=1
LOG=experiments/_sweep.txt
echo "=== sweep started $(date) ===" | tee "$LOG"

run () {
    local name="$1"; shift
    echo "[$(date +%H:%M:%S)] >>> $name" | tee -a "$LOG"
    python -m wildfire.train --exp "sweep_$name" "$@" 2>&1 | tee -a "$LOG"
    echo "[$(date +%H:%M:%S)] <<< $name done" | tee -a "$LOG"
}

# Common: 2K subset, 3 epochs, B=16, focal loss
COMMON="--subset-train 2000 --epochs 3 --batch-size 16 --lr 3e-4 --eval-batches 30"

# A. Reference: uniform (no physics edges) -- our current best at 5K
run A_uniform_h96 --model pignode --uniform-edges --hidden 96 --ode-layers 2 \
    --n-eval-steps 1 --t-end 1.0 $COMMON

# B. uniform + larger model
run B_uniform_h128 --model pignode --uniform-edges --hidden 128 --ode-layers 2 \
    --n-eval-steps 1 --t-end 1.0 $COMMON

# C. uniform + deeper ODE (3 stacked GAT layers)
run C_uniform_3layer --model pignode --uniform-edges --hidden 96 --ode-layers 3 \
    --n-eval-steps 1 --t-end 1.0 $COMMON

# D. uniform + deeper integration (2-step RK4 with t=2)
run D_uniform_2step --model pignode --uniform-edges --hidden 96 --ode-layers 2 \
    --n-eval-steps 2 --t-end 2.0 $COMMON

# E. With physics edges (control) -- expected to underperform if uniform is better
run E_with_edges --model pignode --hidden 96 --ode-layers 2 \
    --n-eval-steps 1 --t-end 1.0 $COMMON

# F. uniform + higher LR (1e-3) -- 3 epochs is short, maybe needs more aggressive LR
run F_uniform_lr1e3 --model pignode --uniform-edges --hidden 96 --ode-layers 2 \
    --n-eval-steps 1 --t-end 1.0 --subset-train 2000 --epochs 3 --batch-size 16 \
    --lr 1e-3 --eval-batches 30

echo "=== sweep done $(date) ===" | tee -a "$LOG"
