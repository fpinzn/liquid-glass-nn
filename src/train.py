"""
Training loop for liquid glass CNN models.
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from tqdm import tqdm

from .models import get_model, count_params, estimate_flops, MODELS
from .dataset import SyntheticGlassDataset, AppleGlassDataset


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        total_loss += loss.item() * inputs.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def log_samples(model, loader, writer, epoch, device, n=4):
    """Log a grid of input / predicted / ground truth to tensorboard."""
    model.eval()
    inputs, targets = next(iter(loader))
    inputs, targets = inputs[:n].to(device), targets[:n].to(device)
    outputs = model(inputs)

    # Clamp to valid range
    outputs = outputs.clamp(0, 1)

    # Build comparison grid: input | predicted | ground truth
    rows = []
    for i in range(n):
        rows.extend([inputs[i].cpu(), outputs[i].cpu(), targets[i].cpu()])

    grid = make_grid(rows, nrow=3, padding=2)
    writer.add_image("samples/input_pred_gt", grid, epoch)


def compute_ssim_batch(pred, target):
    """Simple SSIM approximation for monitoring."""
    # Mean over spatial dims
    mu_p = pred.mean(dim=[2, 3], keepdim=True)
    mu_t = target.mean(dim=[2, 3], keepdim=True)
    sigma_p = ((pred - mu_p) ** 2).mean(dim=[2, 3], keepdim=True)
    sigma_t = ((target - mu_t) ** 2).mean(dim=[2, 3], keepdim=True)
    sigma_pt = ((pred - mu_p) * (target - mu_t)).mean(dim=[2, 3], keepdim=True)

    c1, c2 = 0.01 ** 2, 0.03 ** 2
    ssim = ((2 * mu_p * mu_t + c1) * (2 * sigma_pt + c2)) / (
        (mu_p ** 2 + mu_t ** 2 + c1) * (sigma_p + sigma_t + c2)
    )
    return ssim.mean().item()


def train_model(
    arch: str,
    frames_dir: str,
    dataset_mode: str = "synthetic",
    epochs: int = 200,
    batch_size: int = 16,
    lr: float = 1e-3,
    output_dir: str = "runs",
    surface_params: dict = None,
):
    device = get_device()
    print(f"Device: {device}")

    # Dataset
    if dataset_mode == "apple":
        dataset = AppleGlassDataset(frames_dir, augment=True)
    else:
        dataset = SyntheticGlassDataset(frames_dir, surface_params=surface_params, augment=True)

    # Split 90/10
    n_val = max(1, int(len(dataset) * 0.1))
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)

    # Model
    model = get_model(arch).to(device)
    params = count_params(model)
    flops = estimate_flops(model)
    print(f"Architecture: {arch}")
    print(f"Parameters: {params:,}")
    print(f"Est. FLOPs: {flops:,}")

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.L1Loss()

    # Logging
    run_dir = Path(output_dir) / arch
    run_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(run_dir))

    best_val_loss = float("inf")
    best_epoch = 0

    for epoch in tqdm(range(1, epochs + 1), desc=arch):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val", val_loss, epoch)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

        # SSIM on validation
        if epoch % 10 == 0:
            model.eval()
            ssim_sum, ssim_n = 0, 0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs).clamp(0, 1)
                    ssim_sum += compute_ssim_batch(outputs, targets) * inputs.size(0)
                    ssim_n += inputs.size(0)
            writer.add_scalar("metrics/ssim", ssim_sum / ssim_n, epoch)

        # Log sample images
        if epoch % 20 == 0:
            log_samples(model, val_loader, writer, epoch, device)

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), run_dir / "best.pt")

    # Save final
    torch.save(model.state_dict(), run_dir / "final.pt")

    # Save metadata
    meta = {
        "arch": arch,
        "params": params,
        "flops": flops,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "dataset_mode": dataset_mode,
        "dataset_size": len(dataset),
    }
    with open(run_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    writer.close()
    print(f"  Best val loss: {best_val_loss:.6f} (epoch {best_epoch})")
    return meta


def main():
    parser = argparse.ArgumentParser(description="Train liquid glass CNN")
    parser.add_argument("--arch", type=str, default="all",
                        help=f"Architecture: {list(MODELS.keys())} or 'all'")
    parser.add_argument("--frames-dir", type=str, default="data/frames",
                        help="Directory with input frames (PNG/JPG)")
    parser.add_argument("--dataset-mode", type=str, default="synthetic",
                        choices=["synthetic", "apple"],
                        help="Dataset mode: synthetic (math renderer) or apple (captured pairs)")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output-dir", type=str, default="runs")
    args = parser.parse_args()

    archs = list(MODELS.keys()) if args.arch == "all" else [args.arch]
    results = []

    for arch in archs:
        print(f"\n{'='*50}")
        print(f"Training: {arch}")
        print(f"{'='*50}")
        meta = train_model(
            arch=arch,
            frames_dir=args.frames_dir,
            dataset_mode=args.dataset_mode,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            output_dir=args.output_dir,
        )
        results.append(meta)

    # Print comparison table
    print(f"\n{'='*70}")
    print(f"{'Arch':<15} {'Params':>10} {'FLOPs':>12} {'Val Loss':>10} {'Best Ep':>8}")
    print(f"{'-'*70}")
    for r in sorted(results, key=lambda x: x["best_val_loss"]):
        print(f"{r['arch']:<15} {r['params']:>10,} {r['flops']:>12,} {r['best_val_loss']:>10.6f} {r['best_epoch']:>8}")

    # Save comparison
    with open(Path(args.output_dir) / "comparison.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
