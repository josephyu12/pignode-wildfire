#!/usr/bin/env bash
# All proposal-§6.3 ablations not covered by run_all.sh:
#   #2 feature-group drop:       drop_{topo,weather,fuel,human}
#   #3 4- vs 8-connectivity
#   #4 message-passing depth:    ode_layers ∈ {1,2,3}
#   #5 temporal integration:     {euler, rk4, dopri5} (dopri5 forces CPU)
# Plus the eq. (4)-faithful soft monotonicity training loss and
# the §5 #4 Frobenius dynamics regularizer.
#
# All runs use the same shared subset (5K events, 8 epochs) used by run_all.sh
# so results are directly comparable to the reported PI-GNODE main numbers.
# Each run lands in experiments/<auto-named-dir>/ with metrics.json + log.txt;
# `python -m wildfire.figures` aggregates them into the paper's table/figures.
# NOTE: deliberately NOT using `set -e`. If one ablation crashes (OOM,
# numerical issue, etc.), we still want every other ablation to run so
# `python -m wildfire.figures` can aggregate whatever finished.
cd "$(dirname "$0")/.."
source .venv/bin/activate
export PYTHONUNBUFFERED=1
LOG=experiments/_runlog.txt
mkdir -p experiments

# Fail fast if the data isn't there -- no point running 14 experiments that
# will each error in the same place.
if [ ! -d data/raw/ndws/train.zarr ]; then
    echo "ERROR: data/raw/ndws/train.zarr not found." | tee -a "$LOG"
    echo "Download NDWS first, then re-run.  See README." | tee -a "$LOG"
    exit 1
fi

echo "=== ablations started $(date) ===" >>"$LOG"
PASS=0; FAIL=0; FAILED_RUNS=()

run () {
    local name="$1"; shift
    echo "[$(date +%H:%M:%S)] >>> $name" | tee -a "$LOG"
    if python -m wildfire.train --exp "$name" "$@" 2>&1 | tee -a "$LOG"; then
        echo "[$(date +%H:%M:%S)] <<< $name done" | tee -a "$LOG"
        PASS=$((PASS+1))
    else
        echo "[$(date +%H:%M:%S)] !!! $name FAILED (continuing)" | tee -a "$LOG"
        FAIL=$((FAIL+1))
        FAILED_RUNS+=("$name")
    fi
}

# Common settings (mirror the headline PI-GNODE: h128, ode_layers=2, uniform edges)
SUB="--subset-train 5000 --eval-batches 30"
BASE="--model pignode --uniform-edges --hidden 128 --ode-layers 2 \
    --n-eval-steps 1 --t-end 1.0 --epochs 8 --batch-size 16 --lr 3e-4 $SUB"

# ---------- §6.3 #2: feature-group drop ----------
for grp in topo weather fuel human; do
    run "pignode_drop_${grp}" $BASE --drop-feature-group $grp
done

# ---------- §6.3 #3: 4- vs 8-connectivity ----------
# 8-conn is the headline run; we only need to add 4-conn here.
run pignode_c4 $BASE --connectivity 4

# ---------- §6.3 #4: message-passing depth (ODE function depth) ----------
run pignode_ol1 $BASE --ode-layers 1
# ode_layers=2 is the headline, skip.
run pignode_ol3 $BASE --ode-layers 3

# ---------- §6.3 #5: temporal integration solver ----------
# rk4 is the headline; add Euler (fixed step, MPS-safe) and dopri5 (adaptive).
# dopri5 needs float64 and so currently has to run on CPU.
run pignode_euler  $BASE --solver euler
run pignode_dopri5 --model pignode --uniform-edges --hidden 128 --ode-layers 2 \
    --n-eval-steps 1 --t-end 1.0 --epochs 8 --batch-size 16 --lr 3e-4 $SUB \
    --solver dopri5 --adjoint --device cpu

# ---------- proposal §5 #4: Frobenius dynamics regularization ----------
# Tiny weight to start; if dynamics are well-behaved already we expect a no-op.
run pignode_frob1e-3 $BASE --frobenius-weight 1e-3
run pignode_frob1e-2 $BASE --frobenius-weight 1e-2

# ---------- eq. (4) soft monotonicity training loss ----------
# Differentiable analogue of the burn-irreversibility floor; gives a learning
# signal on burning cells that the inference-only hard floor does not.
run pignode_softmono $BASE --soft-mono-weight 0.1

# ---------- proposal §4.2: do edge features help non-attention GNNs? ----------
# Plain GCN/SAGE don't see edges; the *_edge variants do.
run gcn_edge  --model gcn_edge  --hidden 64 --n-layers 3 --epochs 8 --batch-size 32
run sage_edge --model sage_edge --hidden 64 --n-layers 3 --epochs 8 --batch-size 32

echo "=== ablations done $(date) -- $PASS passed, $FAIL failed ===" | tee -a "$LOG"
if [ $FAIL -gt 0 ]; then
    echo "Failed runs: ${FAILED_RUNS[*]}" | tee -a "$LOG"
fi

# Always aggregate figures from whatever experiments succeeded so even a
# partial completion produces something usable for the paper.
echo "[$(date +%H:%M:%S)] aggregating figures..." | tee -a "$LOG"
python -m wildfire.figures 2>&1 | tee -a "$LOG" || \
    echo "figure aggregation failed (non-fatal)" | tee -a "$LOG"
