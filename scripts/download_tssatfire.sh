#!/usr/bin/env bash
# One-time download + preprocess of TS-SatFire (proposal §2.2).
#
# This is NOT run automatically by run_all.sh because:
#   (a) the raw GeoTIFF dataset is ~71 GB,
#   (b) Kaggle download requires API auth,
#   (c) preprocessing into .npy takes hours.
#
# Run once before any TS-SatFire experiments. Output lands in
# data/raw/tssatfire/{train,val,test}/*.npy and is consumed by
# `wildfire.data.tssatfire.TSSatFireDataset`.
set -e
cd "$(dirname "$0")/.."

DATA_DIR="data/raw/tssatfire"
RAW_DIR="$DATA_DIR/_raw_geotiffs"
REPO_DIR="$DATA_DIR/_official_repo"
TS_LEN=${TS_LEN:-6}
INTERVAL=${INTERVAL:-3}
SPLITS=${SPLITS:-"train val test"}
PYTHON=${PYTHON:-python3}
# Optional: cap how many event IDs per split get preprocessed. The Kaggle
# download is all-or-nothing (~71 GB), but the .npy outputs only include
# events you preprocess. Set MAX_*_EVENTS=10 for a fast sanity-check run.
MAX_TRAIN_EVENTS=${MAX_TRAIN_EVENTS:-0}
MAX_VAL_EVENTS=${MAX_VAL_EVENTS:-0}
MAX_TEST_EVENTS=${MAX_TEST_EVENTS:-0}

mkdir -p "$DATA_DIR"

# 1. Authenticate with Kaggle and download the dataset.
if [ -x ".venv/bin/kaggle" ]; then
    KAGGLE=".venv/bin/kaggle"
elif command -v kaggle >/dev/null 2>&1; then
    KAGGLE="kaggle"
else
    echo "kaggle CLI not found. Install with: pip install kaggle"
    echo "Then put your kaggle.json in ~/.kaggle/kaggle.json (chmod 600)."
    exit 1
fi
mkdir -p "$RAW_DIR"
if [ ! -f "$RAW_DIR/.downloaded" ]; then
    echo "Downloading TS-SatFire from Kaggle (~71 GB)..."
    "$KAGGLE" datasets download -d z789456sx/ts-satfire -p "$RAW_DIR" --unzip
    touch "$RAW_DIR/.downloaded"
fi

# 2. Clone the official repo for its `dataset_gen_pred.py` preprocessor.
if [ ! -d "$REPO_DIR" ]; then
    git clone --depth 1 https://github.com/zhaoyutim/TS-SatFire.git "$REPO_DIR"
fi

# 3. Run the preprocessor for each split. The official script hardcodes a path
#    `/home/z/h/zhao2/CalFireMonitoring/data/`; we override via sed-edit on a
#    copy. (No upstream patch is desirable -- their hardcoded path must change
#    locally regardless.)
read -r -a SPLITS_ARR <<< "$SPLITS"
WORK_DIR="$DATA_DIR/_work"
mkdir -p "$WORK_DIR"
cp -R "$REPO_DIR"/* "$WORK_DIR/"
sed -i.bak "s#/home/z/h/zhao2/CalFireMonitoring/data/#$(realpath $RAW_DIR)/#g" \
    "$WORK_DIR/dataset_gen_pred.py"

# Patch the upstream script to honor MAX_*_EVENTS by truncating its id lists
# right before the locations are selected. We append a small block; this is
# safe because the upstream code's `if __name__ == '__main__':` reads
# train_ids / val_ids / test_ids from module globals.
"$PYTHON" - "$WORK_DIR/dataset_gen_pred.py" "$MAX_TRAIN_EVENTS" "$MAX_VAL_EVENTS" "$MAX_TEST_EVENTS" <<'PY'
import sys, pathlib
path, mt, mv, mte = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
src = pathlib.Path(path).read_text()
inject = f"""
# --- injected by scripts/download_tssatfire.sh ---
if {mt} > 0:
    train_ids = train_ids[:{mt}]
if {mv} > 0:
    val_ids = val_ids[:{mv}]
if {mte} > 0:
    test_ids = test_ids[:{mte}]
    test_label_sel = test_label_sel[:{mte}]
# --- end injection ---
"""
marker = "if __name__ == '__main__':"
assert marker in src, f"upstream layout changed; couldn't find {marker!r}"
src = src.replace(marker, inject + marker)
pathlib.Path(path).write_text(src)
PY

pushd "$WORK_DIR" >/dev/null
for split in "${SPLITS_ARR[@]}"; do
    echo "=== preprocessing TS-SatFire split=$split ==="
    "$PYTHON" dataset_gen_pred.py -mode "$split" -ts "$TS_LEN" -it "$INTERVAL"
done
popd >/dev/null

# 4. Move the .npy outputs into the layout our loader expects.
for split in "${SPLITS_ARR[@]}"; do
    mkdir -p "$DATA_DIR/$split"
    mv "$WORK_DIR"/dataset/dataset_${split}/pred_*_img_seqtoseq*${TS_LEN}i_${INTERVAL}.npy \
       "$DATA_DIR/$split/" 2>/dev/null || true
    mv "$WORK_DIR"/dataset/dataset_${split}/pred_*_label_seqtoseq*${TS_LEN}i_${INTERVAL}.npy \
       "$DATA_DIR/$split/" 2>/dev/null || true
done

echo "Done. TS-SatFire ready at $DATA_DIR/{train,val,test}/."
