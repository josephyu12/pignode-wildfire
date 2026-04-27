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

# ConvAE (Huot et al. replication)
run convae --model convae --epochs 10 --batch-size 32

# Graph baselines
run gcn  --model gcn  --epochs 8 --batch-size 16
run sage --model sage --epochs 8 --batch-size 16
run gat  --model gat  --epochs 8 --batch-size 16

# PI-GNODE main + ablations
run pignode         --model pignode --epochs 8 --batch-size 16
run pignode_no_mono --model pignode --epochs 8 --batch-size 16 --no-monotone
run pignode_uniform --model pignode --epochs 8 --batch-size 16 --uniform-edges

echo "=== run_all done $(date) ===" >>"$LOG"
