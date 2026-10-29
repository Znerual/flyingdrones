#!/bin/bash
# Download MemFlow checkpoints from GitHub releases v1.0.0.
# Place this script in the repo root and run:   bash scripts/download_checkpoints.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CKPT_DIR="$SCRIPT_DIR/../ckpts"
mkdir -p "$CKPT_DIR"

BASE="https://github.com/DQiaole/MemFlow/releases/download/v1.0.0"

CKPTS=(
    MemFlowNet_things.pth
    MemFlowNet_sintel.pth
    MemFlowNet_kitti.pth
    MemFlowNet_spring.pth
    MemFlowNet_T_things.pth
    MemFlowNet_T_things_kitti.pth
    MemFlowNet_T_sintel.pth
    MemFlowNet_T_kitti.pth
)

echo "Downloading MemFlow checkpoints to: $CKPT_DIR"
for ckpt in "${CKPTS[@]}"; do
    dest="$CKPT_DIR/$ckpt"
    if [ -f "$dest" ]; then
        echo "  [skip] $ckpt already exists"
    else
        echo "  Downloading $ckpt ..."
        wget -q --show-progress -O "$dest" "$BASE/$ckpt"
    fi
done
echo "Done. Checkpoints in: $CKPT_DIR"
