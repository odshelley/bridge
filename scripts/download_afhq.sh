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

if [ -d "${DATA_DIR}/afhq/train" ]; then
    echo "AFHQ already present at ${DATA_DIR}/afhq — nothing to do."
    exit 0
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
        count=$(find "${DATA_DIR}/afhq/${split}/${cls}" -type f | wc -l | tr -d ' ')
        echo "  ${split}/${cls}: ${count}"
    done
done
echo "Done. AFHQ is at ${DATA_DIR}/afhq"
