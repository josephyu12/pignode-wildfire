#!/usr/bin/env bash
# Download the NeurIPS 2024 LaTeX style file next to docs/paper.tex.
# The official package ships with the conference template; we mirror it from
# the NeurIPS proceedings repo.
set -e
cd "$(dirname "$0")/.."

DEST="docs/neurips_2024.sty"
URL="https://media.neurips.cc/Conferences/NeurIPS2024/Styles/neurips_2024.sty"

if [ -f "$DEST" ]; then
    echo "$DEST already present, skipping download."
    exit 0
fi

echo "Fetching NeurIPS 2024 style -> $DEST"
if command -v curl >/dev/null 2>&1; then
    curl -fsSL -o "$DEST" "$URL"
elif command -v wget >/dev/null 2>&1; then
    wget -q -O "$DEST" "$URL"
else
    echo "Neither curl nor wget available; download manually:"
    echo "    $URL"
    exit 1
fi

echo "Done. Compile with:  cd docs && pdflatex paper.tex && pdflatex paper.tex"
