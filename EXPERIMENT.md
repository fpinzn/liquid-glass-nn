# NN Liquid Glass Filter — Execution Plan

**Goal:** Find the smallest CNN that approximates a liquid glass refraction effect and runs ≤1ms on iPhone ANE.

**Total estimated time:** 4–6 days of focused work

---

## Phase 1: Ground Truth Renderer (Day 1)

Build an offline Unity shader that produces the "correct" liquid glass effect — this is too expensive for real-time mobile but serves as your training oracle.

**1.1 — Glass geometry & refraction shader** (3–4h)
Write a raymarching or mesh-based refraction shader in Unity that takes a camera frame and outputs the distorted result. Doesn't need to be optimized — it runs offline. You already explored ferrofluid shaders and liquid glass rendering, so you have reference points. A screenspace refraction shader with Snell's law and a normal map is the simplest starting point.

**1.2 — Decompose into displacement + specular passes** (2–3h)
Modify the shader to output three separate render textures: the displacement field (2-channel: dx, dy per pixel), the specular/caustic highlights (1-channel intensity), and the final composited result (for visual validation only). The displacement field is the core training target.

**1.3 — Validation** (1h)
Verify you can reconstruct the composited result from the original frame + displacement + specular by doing the lookup manually in a simple script. If the reconstruction matches, your ground truth pipeline is correct.

**Deliverable:** A Unity scene that takes any input texture and produces displacement + specular ground truth.

---

## Phase 2: Dataset Generation (Day 2, morning)

**2.1 — Collect input frames** (1–2h)
Gather 5,000–10,000 diverse background images. Three sources mixed together: frames captured from your AR camera sessions (the actual use case), a subset of COCO or Open Images (general diversity), and solid colors / gradients / high-frequency textures (stress tests for the displacement field). Resize everything to 256×256.

**2.2 — Batch render ground truth** (2–3h)
Run all input frames through the Unity shader in batch mode (headless). Save each sample as a triplet: `{input.png, displacement.exr, specular.png}`. Use EXR for displacement to preserve float precision. This step is mostly waiting — write the batch script, kick it off, and verify a few random outputs visually.

**2.3 — If glass is animated: vary the glass state** (optional, +2h)
If the glass shape changes (wobble, touch response), generate N variations of the glass normal map or SDF, and render each input frame against K random glass states. This multiplies your dataset. Encode the glass state as additional input channels for training.

**Deliverable:** A dataset folder with paired triplets ready for PyTorch DataLoader.

---

## Phase 3: Training Infrastructure (Day 2, afternoon)

**3.1 — PyTorch dataset + dataloader** (1h)
Standard image-pair dataset. Load input as 3-channel RGB, target as 3-channel (dx, dy, specular). Normalize displacement values to [-1, 1] range.

**3.2 — Model definitions** (1h)
Implement 4–6 architecture variants from the explorer tool as `nn.Module` classes. Include: micro (3-layer/8ch), tiny (5-layer/16ch), small (7-layer/32ch), wide-shallow (2-layer/64ch), bottleneck, and deep-narrow. All fully convolutional with same-padding so output resolution matches input.

**3.3 — Loss function** (1–2h)
Start with L1 loss on the displacement channels + L1 on specular. Optionally add a perceptual loss (VGG feature matching on the *composited* result) to penalize visually wrong distortions. The composite-level loss is important — a small displacement error in a high-frequency region is more visible than in a flat region.

**3.4 — Training loop + logging** (1h)
Standard training loop with wandb or tensorboard logging. Log: loss curves, a grid of sample outputs every N steps (input / predicted displacement / predicted composite / ground truth composite side by side), and per-architecture comparison.

**Deliverable:** A training script that can train any of the 6 architectures with one config flag.

---

## Phase 4: Training Experiments (Day 3)

**4.1 — Train all 6 architectures** (4–6h, mostly waiting)
Each architecture trains for ~200 epochs on 5K–10K samples. On a decent GPU (3060+), expect 20–40 min per architecture. Run them sequentially or in parallel if you have the VRAM.

**4.2 — Evaluate quality vs. size tradeoff** (1–2h)
For each trained model, compute: L1 error on held-out test set, SSIM of composited result vs. ground truth, parameter count, and FLOPs. Build a Pareto chart: x-axis = FLOPs (proxy for latency), y-axis = SSIM. You're looking for the knee of the curve — the smallest model that still looks good.

**4.3 — Ablations if needed** (2–3h)
Based on results, you might want to try: different channel counts between the presets, skip connections (residual), depthwise separable convolutions (fewer FLOPs at same channel width), or output displacement only (drop specular, add it analytically later).

**Deliverable:** A trained model checkpoint for the best 2–3 candidates, plus a comparison table.

---

## Phase 5: CoreML Export & On-Device Benchmark (Day 4)

**5.1 — Export to CoreML** (1–2h)
Use coremltools to convert each candidate. Set compute units to ALL (enables ANE). Use fp16 quantization. Specify ImageType input and MultiArray output. Test that the converted model produces the same outputs as PyTorch (within fp16 tolerance).

**5.2 — Xcode benchmark** (2–3h)
Create a minimal Xcode project or use the CoreML Performance Report in Xcode. Measure actual inference time for each model on a real device. Compare against the explorer estimates. Profile with Instruments to confirm the model is actually dispatching to ANE (not falling back to GPU — common gotcha with certain layer configurations).

**5.3 — ANE compatibility fixes** (1–2h, if needed)
If a model falls back to GPU, identify the offending layer (usually unsupported activations, odd kernel sizes, or specific reshape ops) and replace it with an ANE-friendly alternative. Re-export and re-benchmark.

**Deliverable:** A CoreML model file (.mlpackage) with confirmed ≤1ms ANE inference.

---

## Phase 6: Integration Prototype (Day 5–6)

**6.1 — Camera → model → composite pipeline** (3–4h)
Wire up AVCaptureSession → downsample to 256×256 → run CoreML model → extract displacement + specular → apply displacement lookup on full-res camera texture via Metal → composite specular → display. The displacement lookup and compositing happen in a Metal shader, not the neural network — the NN just predicts the fields.

**6.2 — Metal shader for displacement sampling** (2–3h)
Write a simple Metal compute or fragment shader that reads the displacement field, samples the camera texture at offset coordinates using bilinear interpolation, and adds the specular highlights. This is where the final quality comes together.

**6.3 — End-to-end latency measurement** (1–2h)
Measure the full pipeline latency: capture → resize → NN inference → displacement sample → display. Target: ≤3ms total for the NN + composite step (1ms model + ~2ms for the Metal shader and memory transfers).

**6.4 — Visual polish + edge cases** (2–3h)
Handle edge pixels where displacement goes out of bounds (clamp or mirror), tune specular intensity, test with different lighting conditions and backgrounds. Compare side-by-side with the original Unity shader to validate quality.

**Deliverable:** A working prototype that runs the learned liquid glass effect live on iPhone camera feed.

---

## Risk Factors

**Quality might not converge at the smallest sizes.** The micro (3-layer/8ch) model might not have enough capacity for complex refraction patterns. Mitigation: the wide-shallow architecture tends to work better for spatial effects — prioritize width over depth.

**ANE fallback to GPU.** Certain layer configs silently fall back to GPU, killing performance. Mitigation: stick to vanilla Conv2d + ReLU, avoid GroupNorm, LayerNorm, and exotic activations. Test early in Phase 5.

**Displacement field artifacts at low resolution.** If running the model at 128×128 and upscaling, bilinear interpolation of the displacement field can create visible banding. Mitigation: train at 256×256, or add a bicubic upsampling step.

**Animated glass state increases dataset complexity.** If the glass wobbles or responds to input, you need the model to generalize across glass configurations. Mitigation: start with a static glass shape. Add animation once the static version works.