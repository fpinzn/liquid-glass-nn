"""
Generate diverse synthetic frames for training.

Produces a mix of: solid colors, gradients, noise, checkerboards,
natural-looking textures, and compositions. Designed to prevent
overfitting to any specific content type.
"""

import argparse
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm


def solid_color(size, rng):
    color = rng.randint(0, 256, 3, dtype=np.uint8)
    return np.full((size, size, 3), color, dtype=np.uint8)


def linear_gradient(size, rng):
    c1 = rng.randint(0, 256, 3).astype(np.float32)
    c2 = rng.randint(0, 256, 3).astype(np.float32)
    horizontal = rng.random() > 0.5
    t = np.linspace(0, 1, size, dtype=np.float32)
    if horizontal:
        grad = c1[None, :] + np.outer(t, c2 - c1)
        img = np.tile(grad[:, np.newaxis, :], (1, size, 1))
    else:
        grad = c1[None, :] + np.outer(t, c2 - c1)
        img = np.tile(grad[np.newaxis, :, :], (size, 1, 1))
    return np.clip(img, 0, 255).astype(np.uint8)


def radial_gradient(size, rng):
    c1 = rng.randint(0, 256, 3).astype(np.float32)
    c2 = rng.randint(0, 256, 3).astype(np.float32)
    cx = rng.uniform(0.2, 0.8) * size
    cy = rng.uniform(0.2, 0.8) * size
    y, x = np.mgrid[0:size, 0:size].astype(np.float32)
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    t = np.clip(dist / (size * 0.7), 0, 1)
    img = c1[None, None, :] * (1 - t[:, :, None]) + c2[None, None, :] * t[:, :, None]
    return np.clip(img, 0, 255).astype(np.uint8)


def uniform_noise(size, rng):
    return rng.randint(0, 256, (size, size, 3), dtype=np.uint8)


def smooth_noise(size, rng):
    """Low-frequency colored noise — resembles blurry natural images."""
    small = rng.randint(0, 256, (size // 8, size // 8, 3), dtype=np.uint8)
    return cv2.resize(small, (size, size), interpolation=cv2.INTER_CUBIC)


def perlin_like(size, rng):
    """Multi-octave smooth noise for organic textures."""
    img = np.zeros((size, size, 3), dtype=np.float32)
    for octave in range(4):
        s = max(4, size // (2 ** (octave + 2)))
        noise = rng.randn(s, s, 3).astype(np.float32)
        upsampled = cv2.resize(noise, (size, size), interpolation=cv2.INTER_CUBIC)
        img += upsampled * (0.5 ** octave)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8) * 255
    return img.astype(np.uint8)


def checkerboard(size, rng):
    block = rng.randint(4, 32)
    c1 = rng.randint(0, 256, 3, dtype=np.uint8)
    c2 = rng.randint(0, 256, 3, dtype=np.uint8)
    img = np.zeros((size, size, 3), dtype=np.uint8)
    for y in range(0, size, block):
        for x in range(0, size, block):
            color = c1 if ((y // block + x // block) % 2 == 0) else c2
            img[y:y+block, x:x+block] = color
    return img


def stripes(size, rng):
    freq = rng.randint(4, 40)
    c1 = rng.randint(0, 256, 3, dtype=np.uint8)
    c2 = rng.randint(0, 256, 3, dtype=np.uint8)
    angle = rng.uniform(0, np.pi)
    y, x = np.mgrid[0:size, 0:size].astype(np.float32)
    wave = np.sin((x * np.cos(angle) + y * np.sin(angle)) * 2 * np.pi * freq / size)
    mask = wave > 0
    img = np.where(mask[:, :, None], c1, c2)
    return img.astype(np.uint8)


def concentric_circles(size, rng):
    cx, cy = size // 2, size // 2
    c1 = rng.randint(0, 256, 3, dtype=np.uint8)
    c2 = rng.randint(0, 256, 3, dtype=np.uint8)
    freq = rng.uniform(5, 25)
    y, x = np.mgrid[0:size, 0:size].astype(np.float32)
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    wave = np.sin(dist * 2 * np.pi * freq / size)
    mask = wave > 0
    img = np.where(mask[:, :, None], c1, c2)
    return img.astype(np.uint8)


def composite(size, rng):
    """Layered composition: background + shapes. Mimics real-world scenes."""
    img = linear_gradient(size, rng).astype(np.float32)

    n_shapes = rng.randint(3, 12)
    for _ in range(n_shapes):
        shape_type = rng.choice(["rect", "circle", "line"])
        color = rng.randint(0, 256, 3).tolist()
        if shape_type == "rect":
            x1, y1 = rng.randint(0, size, 2)
            w, h = rng.randint(10, size // 3, 2)
            cv2.rectangle(img, (x1, y1), (x1 + w, y1 + h), color, -1)
        elif shape_type == "circle":
            cx, cy = rng.randint(0, size, 2)
            r = rng.randint(5, size // 4)
            cv2.circle(img, (int(cx), int(cy)), int(r), color, -1)
        elif shape_type == "line":
            x1, y1 = rng.randint(0, size, 2)
            x2, y2 = rng.randint(0, size, 2)
            thickness = rng.randint(1, 8)
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

    # Optional blur
    if rng.random() > 0.5:
        k = rng.choice([3, 5, 7])
        img = cv2.GaussianBlur(img, (k, k), 0)

    return np.clip(img, 0, 255).astype(np.uint8)


def text_like(size, rng):
    """Simulates text/UI content — important for glass-over-UI use case."""
    bg = rng.randint(200, 256, 3, dtype=np.uint8)
    fg = rng.randint(0, 80, 3).tolist()
    img = np.full((size, size, 3), bg, dtype=np.uint8)

    n_lines = rng.randint(5, 20)
    for i in range(n_lines):
        y = int(rng.uniform(0.05, 0.95) * size)
        length = rng.randint(size // 4, size)
        x = rng.randint(0, max(1, size - length))
        thickness = rng.randint(1, 4)
        cv2.line(img, (x, y), (x + length, y), fg, thickness)

    return img


GENERATORS = [
    solid_color,
    linear_gradient,
    radial_gradient,
    uniform_noise,
    smooth_noise,
    perlin_like,
    checkerboard,
    stripes,
    concentric_circles,
    composite,
    text_like,
]

# Weights: favor diverse/complex types, fewer solid colors
WEIGHTS = [0.05, 0.08, 0.08, 0.08, 0.12, 0.12, 0.08, 0.08, 0.06, 0.15, 0.10]


def generate_frames(output_dir: str, n_frames: int = 5000, size: int = 256, seed: int = 42):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(seed)
    weights = np.array(WEIGHTS, dtype=np.float64)
    weights /= weights.sum()

    for i in tqdm(range(n_frames), desc="Generating frames"):
        gen_idx = rng.choice(len(GENERATORS), p=weights)
        img = GENERATORS[gen_idx](size, rng)

        # Random global augmentation
        if rng.random() > 0.7:
            # Random rotation
            angle = rng.uniform(-30, 30)
            M = cv2.getRotationMatrix2D((size / 2, size / 2), angle, 1)
            img = cv2.warpAffine(img, M, (size, size), borderMode=cv2.BORDER_REFLECT)

        cv2.imwrite(str(output_dir / f"frame_{i:05d}.png"), img)

    print(f"Generated {n_frames} frames in {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="data/frames")
    parser.add_argument("--n-frames", type=int, default=5000)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    generate_frames(args.output_dir, args.n_frames, args.size, args.seed)


if __name__ == "__main__":
    main()
