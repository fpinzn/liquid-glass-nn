"""
Dataset classes for liquid glass training.

Supports two modes:
1. Synthetic: generates ground truth on-the-fly from video frames + math renderer
2. Apple: loads pre-captured input/composited pairs from the iOS capture app
"""

import json
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from pathlib import Path
from .ground_truth import generate_ground_truth


class SyntheticGlassDataset(Dataset):
    """Dataset that generates glass effect on-the-fly from extracted video frames."""

    def __init__(
        self,
        frames_dir: str,
        size: int = 256,
        surface_params: dict = None,
        augment: bool = True,
    ):
        self.frames_dir = Path(frames_dir)
        self.size = size
        self.surface_params = surface_params or {}
        self.augment = augment

        self.frame_paths = sorted(self.frames_dir.glob("*.png"))
        if not self.frame_paths:
            self.frame_paths = sorted(self.frames_dir.glob("*.jpg"))
        if not self.frame_paths:
            raise ValueError(f"No frames found in {frames_dir}")

    def __len__(self):
        return len(self.frame_paths)

    def __getitem__(self, idx):
        # Load frame
        img = cv2.imread(str(self.frame_paths[idx]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.size, self.size))

        # Augmentation
        if self.augment:
            if np.random.random() > 0.5:
                img = np.fliplr(img).copy()
            if np.random.random() > 0.5:
                img = np.flipud(img).copy()
            # Random brightness/contrast
            if np.random.random() > 0.5:
                alpha = np.random.uniform(0.8, 1.2)
                beta = np.random.uniform(-20, 20)
                img = np.clip(img * alpha + beta, 0, 255).astype(np.uint8)

        # Generate ground truth
        gt = generate_ground_truth(img, self.surface_params)

        # Convert to tensors: [C, H, W], normalized to [0, 1] for images
        input_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        target_tensor = torch.from_numpy(gt["composited"]).permute(2, 0, 1).float() / 255.0

        return input_tensor, target_tensor


class AppleGlassDataset(Dataset):
    """Dataset from iOS capture app: paired input/composited PNGs."""

    def __init__(
        self,
        dataset_dir: str,
        size: int = 256,
        augment: bool = True,
    ):
        self.dataset_dir = Path(dataset_dir)
        self.size = size
        self.augment = augment

        # Find all input files
        self.input_paths = sorted(self.dataset_dir.glob("input_*.png"))
        if not self.input_paths:
            raise ValueError(f"No input_*.png files found in {dataset_dir}")

        # Verify composited files exist
        for p in self.input_paths:
            comp = p.parent / p.name.replace("input_", "composited_")
            if not comp.exists():
                raise ValueError(f"Missing composited file: {comp}")

        # Load manifest if available
        manifest_path = self.dataset_dir / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path) as f:
                self.manifest = json.load(f)
        else:
            self.manifest = {}

    def __len__(self):
        return len(self.input_paths)

    def __getitem__(self, idx):
        input_path = self.input_paths[idx]
        comp_path = input_path.parent / input_path.name.replace("input_", "composited_")

        inp = cv2.imread(str(input_path))
        inp = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)
        inp = cv2.resize(inp, (self.size, self.size))

        comp = cv2.imread(str(comp_path))
        comp = cv2.cvtColor(comp, cv2.COLOR_BGR2RGB)
        comp = cv2.resize(comp, (self.size, self.size))

        # Augmentation — apply same transform to both
        if self.augment:
            if np.random.random() > 0.5:
                inp = np.fliplr(inp).copy()
                comp = np.fliplr(comp).copy()
            if np.random.random() > 0.5:
                inp = np.flipud(inp).copy()
                comp = np.flipud(comp).copy()
            if np.random.random() > 0.5:
                alpha = np.random.uniform(0.8, 1.2)
                beta = np.random.uniform(-20, 20)
                inp = np.clip(inp * alpha + beta, 0, 255).astype(np.uint8)
                comp = np.clip(comp * alpha + beta, 0, 255).astype(np.uint8)

        input_tensor = torch.from_numpy(inp).permute(2, 0, 1).float() / 255.0
        target_tensor = torch.from_numpy(comp).permute(2, 0, 1).float() / 255.0

        return input_tensor, target_tensor
