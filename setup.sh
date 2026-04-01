#!/bin/bash
set -e

echo "=== Liquid Glass NN — Setup ==="

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

echo "Installing dependencies..."
source venv/bin/activate
pip install -q --upgrade pip
pip install -q -r requirements.txt

echo ""
echo "Setup complete. To activate: source venv/bin/activate"
echo ""
echo "Next steps:"
echo "  1. Drop your video(s) in data/video/"
echo "  2. Run: ./train.sh"
