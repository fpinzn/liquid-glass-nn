#!/bin/bash
set -e

echo "=== Liquid Glass NN — Setup ==="

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

echo "Installing dependencies..."
source venv/bin/activate 2>/dev/null || source venv/Scripts/activate

if [ -f "venv/Scripts/python.exe" ]; then
    PYTHON=venv/Scripts/python.exe
else
    PYTHON=venv/bin/python
fi
$PYTHON -m pip install -q --upgrade pip
$PYTHON -m pip install -q -r requirements.txt

echo ""
echo "Setup complete. To activate: source venv/bin/activate 2>/dev/null || source venv/Scripts/activate"
echo ""
echo "Next steps:"
echo "  1. Drop your video(s) in data/video/"
echo "  2. Run: ./train.sh"
