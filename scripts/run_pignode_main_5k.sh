#!/usr/bin/env bash
# Wait for the current uniform-ablation run to finish, then train PI-GNODE main
# on the 5K subset (h96, ode_layers=2, monotonicity inference-only, physics
# edges on, 8 epochs) -- the missing reference cell for clean ablation deltas.
set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate

LOG=experiments/_runlog.txt
echo "=== run_pignode_main_5k waiter started $(date) ===" >>"$LOG"

# Wait for uniform run to finish (its log line "<<< pignode_uniform done")
until grep -q "<<< pignode_uniform done" "$LOG" 2>/dev/null; do
    sleep 30
done

echo "[$(date +%H:%M:%S)] >>> pignode_main_5k" | tee -a "$LOG"
python -m wildfire.train --exp pignode_main_5k --model pignode \
    --hidden 96 --ode-layers 2 --n-eval-steps 1 --t-end 1.0 \
    --epochs 8 --batch-size 16 --lr 3e-4 \
    --subset-train 5000 --eval-batches 30 2>&1 | tee -a "$LOG"
echo "[$(date +%H:%M:%S)] <<< pignode_main_5k done" | tee -a "$LOG"
