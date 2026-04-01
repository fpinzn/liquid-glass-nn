"""
Generate an HTML report comparing all trained architectures side-by-side.

Produces: results/report.html with embedded base64 images.
Single self-contained file — no external dependencies to view.
"""

import argparse
import base64
import json
import io
from pathlib import Path

import numpy as np
import cv2
import torch
from PIL import Image

from .models import get_model, count_params, estimate_flops
from .dataset import SyntheticGlassDataset
from .ground_truth import generate_ground_truth


def tensor_to_base64(tensor):
    """Convert a [C, H, W] tensor to base64 PNG string."""
    img = (tensor.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    pil = Image.fromarray(img)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def img_tag(b64, label=""):
    return f'<div class="img-cell"><img src="data:image/png;base64,{b64}"/><span>{label}</span></div>'


def generate_report(
    runs_dir: str = "runs",
    frames_dir: str = "data/frames",
    output_dir: str = "results",
    n_samples: int = 8,
    device_str: str = None,
):
    runs_dir = Path(runs_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if device_str:
        device = torch.device(device_str)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Find trained architectures
    arch_dirs = sorted([d for d in runs_dir.iterdir() if d.is_dir() and (d / "best.pt").exists()])
    if not arch_dirs:
        print("No trained models found in runs/")
        return

    # Load comparison data
    comparison_path = runs_dir / "comparison.json"
    if comparison_path.exists():
        with open(comparison_path) as f:
            comparison = {r["arch"]: r for r in json.load(f)}
    else:
        comparison = {}

    # Load sample images
    dataset = SyntheticGlassDataset(frames_dir, augment=False)
    indices = np.linspace(0, len(dataset) - 1, n_samples, dtype=int)

    # Load all models
    models = {}
    for arch_dir in arch_dirs:
        arch = arch_dir.name
        try:
            model = get_model(arch)
            model.load_state_dict(torch.load(arch_dir / "best.pt", map_location=device, weights_only=True))
            model.to(device).eval()
            models[arch] = model
        except Exception as e:
            print(f"Skipping {arch}: {e}")

    print(f"Loaded {len(models)} models: {list(models.keys())}")

    # Generate predictions
    rows_html = []
    with torch.no_grad():
        for idx in indices:
            inp, target = dataset[idx]
            inp_dev = inp.unsqueeze(0).to(device)

            inp_b64 = tensor_to_base64(inp)
            gt_b64 = tensor_to_base64(target)

            row = img_tag(inp_b64, "Input") + img_tag(gt_b64, "Ground Truth")

            for arch, model in models.items():
                out = model(inp_dev).squeeze(0).cpu()
                out_b64 = tensor_to_base64(out)
                row += img_tag(out_b64, arch)

            rows_html.append(f'<div class="row">{row}</div>')

    # Build metrics table
    metrics_rows = ""
    for arch in models:
        meta = comparison.get(arch, {})
        model = models[arch]
        params = count_params(model)
        flops = estimate_flops(model)
        val_loss = meta.get("best_val_loss", "N/A")
        best_ep = meta.get("best_epoch", "N/A")
        val_str = f"{val_loss:.6f}" if isinstance(val_loss, float) else val_loss
        metrics_rows += f"""
        <tr>
            <td>{arch}</td>
            <td>{params:,}</td>
            <td>{flops:,}</td>
            <td>{val_str}</td>
            <td>{best_ep}</td>
        </tr>"""

    # Column headers
    col_headers = "<span>Input</span><span>Ground Truth</span>"
    for arch in models:
        col_headers += f"<span>{arch}</span>"

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Liquid Glass NN — Training Report</title>
<style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #111; color: #eee; padding: 24px; }}
    h1 {{ font-size: 24px; margin-bottom: 8px; }}
    h2 {{ font-size: 18px; margin: 32px 0 12px; color: #aaa; }}
    .subtitle {{ color: #888; margin-bottom: 32px; }}
    table {{ border-collapse: collapse; margin: 16px 0; }}
    th, td {{ padding: 8px 16px; text-align: right; border: 1px solid #333; }}
    th {{ background: #222; color: #aaa; text-align: left; }}
    td:first-child {{ text-align: left; font-weight: 600; }}
    tr:hover {{ background: #1a1a1a; }}
    .best {{ color: #4f4; font-weight: bold; }}
    .col-headers {{ display: flex; gap: 4px; margin-bottom: 4px; }}
    .col-headers span {{
        width: 256px; text-align: center; font-size: 12px; color: #888;
        font-weight: 600; text-transform: uppercase; letter-spacing: 1px;
    }}
    .row {{ display: flex; gap: 4px; margin-bottom: 12px; }}
    .img-cell {{ display: flex; flex-direction: column; align-items: center; }}
    .img-cell img {{ width: 256px; height: 256px; image-rendering: pixelated; border: 1px solid #333; }}
    .img-cell span {{ font-size: 11px; color: #666; margin-top: 2px; }}
    .samples {{ overflow-x: auto; }}
</style>
</head>
<body>
    <h1>Liquid Glass NN — Training Report</h1>
    <p class="subtitle">Comparing {len(models)} architectures on {n_samples} sample images</p>

    <h2>Metrics</h2>
    <table>
        <tr><th>Architecture</th><th>Parameters</th><th>FLOPs</th><th>Val Loss (L1)</th><th>Best Epoch</th></tr>
        {metrics_rows}
    </table>

    <h2>Side-by-Side Comparison</h2>
    <div class="samples">
        <div class="col-headers">{col_headers}</div>
        {"".join(rows_html)}
    </div>
</body>
</html>"""

    report_path = output_dir / "report.html"
    with open(report_path, "w") as f:
        f.write(html)

    print(f"Report saved to {report_path}")
    return report_path


def main():
    parser = argparse.ArgumentParser(description="Generate HTML comparison report")
    parser.add_argument("--runs-dir", type=str, default="runs")
    parser.add_argument("--frames-dir", type=str, default="data/frames")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--n-samples", type=int, default=8)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    generate_report(
        runs_dir=args.runs_dir,
        frames_dir=args.frames_dir,
        output_dir=args.output_dir,
        n_samples=args.n_samples,
        device_str=args.device,
    )


if __name__ == "__main__":
    main()
