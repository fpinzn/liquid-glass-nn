#!/bin/bash
set -e

echo "=== Liquid Glass NN — Train ==="

# Activate venv
if [ ! -d "venv" ]; then
    echo "Run ./setup.sh first"
    exit 1
fi
source venv/bin/activate

# Step 1: Extract frames from video
if [ ! -d "data/frames" ] || [ -z "$(ls data/frames/ 2>/dev/null)" ]; then
    if [ -z "$(ls data/video/ 2>/dev/null)" ]; then
        echo "ERROR: No videos found in data/video/"
        echo "Drop your video file(s) there and re-run."
        exit 1
    fi
    echo ""
    echo "--- Extracting frames from video ---"
    python -m src.extract_frames --video data/video/ --output-dir data/frames --size 256
else
    echo "Frames already extracted ($(ls data/frames/*.png 2>/dev/null | wc -l) frames)"
fi

# Step 2: Train all architectures
echo ""
echo "--- Training all architectures ---"
echo "Tensorboard: tensorboard --logdir runs/"
echo ""

python -m src.train \
    --arch all \
    --frames-dir data/frames \
    --dataset-mode synthetic \
    --epochs 200 \
    --batch-size 16 \
    --lr 1e-3 \
    --output-dir runs

echo ""
echo "=== Training complete ==="
echo "Results: runs/comparison.json"
echo "Checkpoints: runs/<arch>/best.pt"
echo "Tensorboard: tensorboard --logdir runs/"
