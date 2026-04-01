#!/bin/bash
set -e

echo "=== Liquid Glass NN — Train ==="

# Auto-setup if needed
if [ ! -d "venv" ]; then
    echo "--- First run: setting up environment ---"
    ./setup.sh
fi
source venv/bin/activate

# Step 1: Get training frames
if [ ! -d "data/frames" ] || [ -z "$(ls data/frames/ 2>/dev/null)" ]; then
    if [ -n "$(ls data/video/ 2>/dev/null)" ]; then
        echo ""
        echo "--- Extracting frames from video ---"
        python -m src.extract_frames --video data/video/ --output-dir data/frames --size 256
    else
        echo ""
        echo "--- No video found, generating synthetic frames ---"
        python -m src.generate_frames --output-dir data/frames --n-frames 5000 --size 256
    fi
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
