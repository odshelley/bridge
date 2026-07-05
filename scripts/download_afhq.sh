#!/usr/bin/env bash
# Download AFHQ (Animal Faces-HQ) as released with StarGAN v2 (Choi et al. 2020).
# Usage: bash scripts/download_afhq.sh [DATA_DIR]   (default: data)
#
# If the Dropbox URL below has rotted, alternatives:
#   - StarGAN v2 repo: https://github.com/clovaai/stargan-v2 (download.sh, target afhq-dataset)
#   - HuggingFace: https://huggingface.co/datasets/huggan/AFHQ
#   - Kaggle: search "afhq"
set -euo pipefail

DATA_DIR="${1:-data}"
URL="https://www.dropbox.com/s/t9l9o3vsx2jai3z/afhq.zip?dl=1"
ZIP_PATH="${DATA_DIR}/afhq.zip"

# AFHQ counts as present only if all six class directories exist and are non-empty.
afhq_complete() {
    local split cls dir
    for split in train val; do
        for cls in cat dog wild; do
            dir="${DATA_DIR}/afhq/${split}/${cls}"
            [ -d "${dir}" ] || return 1
            [ -n "$(find "${dir}" -type f -print -quit)" ] || return 1
        done
    done
    return 0
}

if afhq_complete; then
    echo "AFHQ already present at ${DATA_DIR}/afhq — nothing to do."
    exit 0
elif [ -d "${DATA_DIR}/afhq" ]; then
    echo "ERROR: ${DATA_DIR}/afhq exists but the tree looks partial (expected non-empty" >&2
    echo "{train,val}/{cat,dog,wild}). Remove ${DATA_DIR}/afhq and re-run this script." >&2
    exit 1
fi

mkdir -p "${DATA_DIR}"
echo "Downloading AFHQ (~500 MB)..."
curl -L --fail -o "${ZIP_PATH}" "${URL}"

echo "Unzipping..."
unzip -q "${ZIP_PATH}" -d "${DATA_DIR}"
rm "${ZIP_PATH}"

echo "Image counts:"
for split in train val; do
    for cls in cat dog wild; do
        dir="${DATA_DIR}/afhq/${split}/${cls}"
        if [ ! -d "${dir}" ]; then
            echo "ERROR: expected ${split}/${cls} not found after unzip — the archive layout" >&2
            echo "may have changed; see fallback URLs in this script." >&2
            exit 1
        fi
        count=$(find "${dir}" -type f | wc -l | tr -d ' ')
        echo "  ${split}/${cls}: ${count}"
        if [ "${count}" -eq 0 ]; then
            echo "  WARNING: ${split}/${cls} is empty" >&2
        fi
    done
done
echo "Done. AFHQ is at ${DATA_DIR}/afhq"
