# Liquid Glass NN

Train a CNN to replicate a liquid glass refraction effect. Compares 7 architectures from tiny (~1K params) to U-Net (~31M params) to find the best quality/size tradeoff.

## Quick Start

```bash
git clone https://github.com/fpinzn/liquid-glass-nn.git
cd liquid-glass-nn
./train.sh
```

That's it. The script handles everything:
1. Creates a Python venv and installs dependencies
2. Generates 5000 diverse synthetic training frames
3. Trains all 7 architectures (200 epochs each)
4. Generates an HTML report with side-by-side comparisons
5. Commits and pushes results back to this repo

## Requirements

- Python 3.10+
- A GPU is recommended (CUDA or Apple MPS). CPU works but is slow.

### Estimated training time (all 7 architectures)

| Hardware | Time |
|----------|------|
| A100 / 4090 | ~1 hour |
| 3060 / 3070 | ~2 hours |
| Apple M2 (MPS) | ~5 hours |
| CPU only | ~12+ hours |

## Architectures

| Name | Params | Description |
|------|--------|-------------|
| micro | 1K | 3-layer, 8ch — smallest possible |
| tiny | 8K | 5-layer, 16ch |
| small | 48K | 7-layer, 32ch |
| wide_shallow | 4K | 2-layer, 64ch — width over depth |
| bottleneck | 19K | Encoder-decoder with skip connections |
| deep_narrow | 5K | 9-layer, 8ch — depth over width |
| unet | 31M | Full U-Net — quality ceiling |

## Results

After training, check:
- `results/report.html` — visual side-by-side comparison
- `runs/comparison.json` — metrics for all architectures
- `runs/<arch>/best.pt` — best checkpoint per architecture

To view tensorboard logs:
```bash
source venv/bin/activate
tensorboard --logdir runs/
```

## Using your own video

Drop a `.mp4` or `.mov` file in `data/video/` before running `./train.sh`. Frames will be extracted automatically instead of generating synthetic ones.

## Training a single architecture

```bash
source venv/bin/activate
python -m src.train --arch unet --frames-dir data/frames --epochs 200
```
