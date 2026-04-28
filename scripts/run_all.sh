#!/usr/bin/env bash
# Train every model sequentially. Each run is a fresh Python process so MPS state is clean.
# Logs go to experiments/<exp>/log.txt and the global experiments/_runlog.txt.
set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate

export PYTHONUNBUFFERED=1
LOG=experiments/_runlog.txt
mkdir -p experiments
echo "=== run_all started $(date) ===" >>"$LOG"

run () {
    local name="$1"; shift
    echo "[$(date +%H:%M:%S)] >>> $name" | tee -a "$LOG"
    python -m wildfire.train --exp "$name" "$@" 2>&1 | tee -a "$LOG"
    echo "[$(date +%H:%M:%S)] <<< $name done" | tee -a "$LOG"
}

# Traditional baselines (sklearn)
echo "[$(date +%H:%M:%S)] >>> LR + RF" | tee -a "$LOG"
python -m wildfire.baselines_run 2>&1 | tee -a "$LOG"
echo "[$(date +%H:%M:%S)] <<< LR + RF done" | tee -a "$LOG"

# Fast baselines on FULL train (in-memory => no I/O bottleneck)
run convae --model convae --epochs 12 --batch-size 64
run gcn    --model gcn    --epochs 8  --batch-size 32
run sage   --model sage   --epochs 8  --batch-size 32

# GAT and PI-GNODE on 5K-event subsample (compute-bound on M1).
# Eval on full eval (1877) capped at 30 batches per epoch for speed; final test on full test.
SUB="--subset-train 5000 --eval-batches 30"

run gat  --model gat  --epochs 8 --batch-size 32 $SUB

# PI-GNODE main + ablations.
# t_end=2.0 with 2-step RK4 = 8 NFE -> deeper effective message-passing than 1-step.
# LR 1e-3 (3x higher) since model was undertrained at 3e-4.
PIG="--model pignode --epochs 8 --batch-size 16 --n-eval-steps 2 --t-end 2.0 --lr 1e-3 $SUB"
run pignode         $PIG
run pignode_no_mono $PIG --no-monotone
run pignode_uniform $PIG --uniform-edges

echo "=== run_all done $(date) ===" >>"$LOG"
