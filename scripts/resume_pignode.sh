#!/usr/bin/env bash
# Resume training:
#  (1) pignode_main_5k -- the missing reference cell for the ablation grid
#  (2) pignode_uniform_full -- best variant (no physics edges) on FULL data,
#      since uniform won on the 5K subset (0.220 > GAT 0.218); pushing for the
#      paper headline number.
set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate

export PYTHONUNBUFFERED=1
LOG=experiments/_runlog.txt
echo "=== resume_pignode started $(date) ===" >>"$LOG"

run () {
    local name="$1"; shift
    echo "[$(date +%H:%M:%S)] >>> $name" | tee -a "$LOG"
    python -m wildfire.train --exp "$name" "$@" 2>&1 | tee -a "$LOG"
    echo "[$(date +%H:%M:%S)] <<< $name done" | tee -a "$LOG"
}

ARCH="--hidden 96 --ode-layers 2 --n-eval-steps 1 --t-end 1.0 --batch-size 16 --lr 3e-4"

# (1) Main on 5K (with monotonicity inf-only + physics edges)
run pignode_main_5k --model pignode --epochs 8 \
    --subset-train 5000 --eval-batches 30 $ARCH

# (2) Uniform on FULL data (no physics edges, with monotonicity inf-only)
run pignode_uniform_full --model pignode --epochs 6 --uniform-edges $ARCH

echo "=== resume_pignode done $(date) ===" >>"$LOG"
