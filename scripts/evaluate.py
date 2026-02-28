"""
Offline Evaluation Script for Z-Image World Model

Measures real-time performance and generation quality metrics:

1. FPS benchmark (inference speed)
2. Temporal consistency (warp error and LPIPS between consecutive frames)
3. Action responsiveness (output divergence for different actions)
4. PSNR/SSIM (when ground-truth reference frames are available)

Usage:
    # Benchmark FPS only (no GPU-heavy metrics)
    python scripts/evaluate.py --mode fps --model_path checkpoints/model.pt

    # Full quality evaluation
    python scripts/evaluate.py --mode full --model_path checkpoints/model.pt \
        --video_path data/videos/test.mp4

    # Action responsiveness test
    python scripts/evaluate.py --mode action --model_path checkpoints/model.pt
"""

import argparse
import os
import sys
import time
import math
from pathlib import Path
from typing import Optional

# Allow importing from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

# Fix protobuf/onnx incompatibility with system torch
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Metric implementations
# ──────────────────────────────────────────────────────────────────────────────

def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Peak Signal-to-Noise Ratio between two image tensors.

    Args:
        pred: Predicted images (B, C, H, W) or (B, F, C, H, W) in [0, 1]
        target: Reference images, same shape

    Returns:
        Mean PSNR in dB
    """
    mse = F.mse_loss(pred, target).item()
    if mse < 1e-10:
        return float("inf")
    return 20.0 * math.log10(1.0 / math.sqrt(mse))


def compute_ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Structural Similarity Index between image tensors.

    Simplified implementation using luminance, contrast, and structure terms.
    For production evaluation use skimage.metrics.structural_similarity.

    Args:
        pred: (B, C, H, W) or (B, F, C, H, W) in [0, 1]
        target: Reference, same shape

    Returns:
        Mean SSIM in [0, 1]
    """
    # Flatten batch/frame dims
    if pred.dim() == 5:
        b, f = pred.shape[:2]
        pred = pred.reshape(b * f, *pred.shape[2:])
        target = target.reshape(b * f, *target.shape[2:])

    # Gaussian kernel for local statistics
    k1, k2 = 0.01, 0.03
    C1, C2 = (k1 ** 2), (k2 ** 2)

    mu1 = F.avg_pool2d(pred, kernel_size=11, stride=1, padding=5)
    mu2 = F.avg_pool2d(target, kernel_size=11, stride=1, padding=5)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.avg_pool2d(pred ** 2, 11, 1, 5) - mu1_sq
    sigma2_sq = F.avg_pool2d(target ** 2, 11, 1, 5) - mu2_sq
    sigma12 = F.avg_pool2d(pred * target, 11, 1, 5) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return ssim_map.mean().item()


def compute_temporal_consistency(frames: torch.Tensor) -> dict[str, float]:
    """Measure temporal consistency between consecutive frames.

    Metrics:
    - L2 pixel distance (frame-to-frame)
    - LPIPS perceptual distance (proxy: mean absolute difference after VGG-like transform)

    Note: True warp consistency requires optical flow estimation.
    This uses a simpler frame-diff proxy.

    Args:
        frames: (T, C, H, W) or (B, T, C, H, W) in [0, 1]

    Returns:
        Dictionary with consistency metrics
    """
    if frames.dim() == 5:
        b, t, c, h, w = frames.shape
        frames = frames.reshape(b * t, c, h, w)

    if frames.shape[0] < 2:
        return {"frame_l2": 0.0, "frame_lpips_proxy": 0.0}

    consecutive_l2 = []
    consecutive_lpips = []

    for i in range(len(frames) - 1):
        f1 = frames[i]
        f2 = frames[i + 1]

        # L2 distance
        l2 = F.mse_loss(f1, f2).item()
        consecutive_l2.append(l2)

        # LPIPS proxy: mean absolute difference after edge detection
        # (edges are perceptually important)
        f1_edge = _edge_magnitude(f1.unsqueeze(0))
        f2_edge = _edge_magnitude(f2.unsqueeze(0))
        lpips_proxy = F.l1_loss(f1_edge, f2_edge).item()
        consecutive_lpips.append(lpips_proxy)

    return {
        "frame_l2": float(np.mean(consecutive_l2)),
        "frame_lpips_proxy": float(np.mean(consecutive_lpips)),
        "frame_l2_std": float(np.std(consecutive_l2)),
    }


def _edge_magnitude(x: torch.Tensor) -> torch.Tensor:
    """Simple Sobel edge magnitude for LPIPS proxy."""
    # Grayscale
    gray = x.mean(dim=1, keepdim=True)

    # Sobel kernels
    sobel_x = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        dtype=x.dtype, device=x.device,
    ).view(1, 1, 3, 3)
    sobel_y = torch.tensor(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
        dtype=x.dtype, device=x.device,
    ).view(1, 1, 3, 3)

    gx = F.conv2d(gray, sobel_x, padding=1)
    gy = F.conv2d(gray, sobel_y, padding=1)
    return torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)


def compute_action_responsiveness(
    pipeline,
    initial_frame: torch.Tensor,
    num_frames: int = 16,
    device: torch.device = torch.device("cuda"),
) -> dict[str, float]:
    """Measure how much different actions change the output.

    Runs rollouts from the same initial frame with different action sequences
    and measures divergence between outputs.

    Args:
        pipeline: Inference pipeline with .set_initial_frame() and .step()
        initial_frame: Starting frame (1, 3, H, W) in [0, 1]
        num_frames: Number of frames to generate per rollout
        device: Target device

    Returns:
        Dict with action responsiveness metrics
    """
    from models import ActionSpace

    action_pairs = [
        (ActionSpace.IDLE, ActionSpace.MOVE_FORWARD, "idle_vs_forward"),
        (ActionSpace.MOVE_FORWARD, ActionSpace.MOVE_BACKWARD, "forward_vs_backward"),
        (ActionSpace.MOVE_LEFT, ActionSpace.MOVE_RIGHT, "left_vs_right"),
    ]

    results = {}

    for action_a, action_b, name in action_pairs:
        # Rollout A
        pipeline.set_initial_frame(initial_frame.clone())
        frames_a = []
        for _ in range(num_frames):
            with torch.inference_mode():
                frame = pipeline.step(action_a)
            frames_a.append(frame)
        frames_a = torch.cat(frames_a, dim=0)

        # Rollout B (same start, different action)
        pipeline.set_initial_frame(initial_frame.clone())
        frames_b = []
        for _ in range(num_frames):
            with torch.inference_mode():
                frame = pipeline.step(action_b)
            frames_b.append(frame)
        frames_b = torch.cat(frames_b, dim=0)

        # Compute divergence
        divergence = F.mse_loss(frames_a, frames_b).item()
        l1_divergence = F.l1_loss(frames_a, frames_b).item()

        results[f"{name}_mse"] = divergence
        results[f"{name}_l1"] = l1_divergence

    # Overall responsiveness score: average across pairs
    mse_values = [v for k, v in results.items() if k.endswith("_mse")]
    results["mean_action_divergence"] = float(np.mean(mse_values)) if mse_values else 0.0

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Benchmark functions
# ──────────────────────────────────────────────────────────────────────────────

def benchmark_fps(
    pipeline,
    num_frames: int = 100,
    warmup_frames: int = 10,
    device: torch.device = torch.device("cuda"),
) -> dict[str, float]:
    """Benchmark inference FPS.

    Args:
        pipeline: Inference pipeline with .step() method
        num_frames: Number of frames to benchmark
        warmup_frames: Warmup frames (not counted)

    Returns:
        Dict with fps, latency_ms, and breakdown metrics
    """
    from models import ActionSpace

    dummy_frame = torch.rand(1, 3, 128, 128, device=device)
    pipeline.set_initial_frame(dummy_frame)

    # Warmup
    print(f"  Warming up ({warmup_frames} frames)...")
    for _ in range(warmup_frames):
        pipeline.step(ActionSpace.IDLE)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Benchmark
    print(f"  Benchmarking ({num_frames} frames)...")
    start = time.perf_counter()

    for i in range(num_frames):
        action = ActionSpace(i % ActionSpace.num_actions())
        pipeline.step(action)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start
    fps = num_frames / elapsed
    latency_ms = elapsed / num_frames * 1000

    return {
        "fps": fps,
        "latency_ms": latency_ms,
        "total_frames": num_frames,
        "total_time_s": elapsed,
    }


def evaluate_generation_quality(
    pipeline,
    video_path: str,
    device: torch.device = torch.device("cuda"),
) -> dict[str, float]:
    """Evaluate generation quality against reference video.

    Loads a video clip, conditions on the first frame + true actions
    (or IDLE if no action file), generates subsequent frames, and
    compares to ground truth.

    Args:
        pipeline: Inference pipeline
        video_path: Path to reference video file (.mp4, .avi, etc.)
        device: Target device

    Returns:
        Dict with PSNR, SSIM, temporal consistency metrics
    """
    try:
        import cv2
    except ImportError:
        print("Warning: OpenCV not available, skipping video quality evaluation")
        return {}

    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret or len(frames) >= 32:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (128, 128))
        frames.append(torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0)
    cap.release()

    if len(frames) < 2:
        print(f"Warning: Could not read frames from {video_path}")
        return {}

    frames = torch.stack(frames, dim=0)  # (T, 3, H, W)

    # Condition on first frame, generate rest
    pipeline.set_initial_frame(frames[0:1].to(device))
    generated = [frames[0:1]]

    from models import ActionSpace
    for _ in range(len(frames) - 1):
        with torch.inference_mode():
            frame = pipeline.step(ActionSpace.IDLE).cpu()
        generated.append(frame)

    generated = torch.cat(generated, dim=0)  # (T, 3, H, W)
    reference = frames  # (T, 3, H, W)

    # PSNR and SSIM (skip first frame, it's conditioning)
    psnr = compute_psnr(generated[1:].to(device), reference[1:].to(device))
    ssim = compute_ssim(generated[1:].to(device), reference[1:].to(device))

    # Temporal consistency of generated frames
    gen_consistency = compute_temporal_consistency(generated.to(device))

    # Temporal consistency of reference (ground truth)
    ref_consistency = compute_temporal_consistency(reference.to(device))

    return {
        "psnr_db": psnr,
        "ssim": ssim,
        "gen_frame_l2": gen_consistency["frame_l2"],
        "ref_frame_l2": ref_consistency["frame_l2"],
        "temporal_consistency_ratio": (
            ref_consistency["frame_l2"] / max(gen_consistency["frame_l2"], 1e-8)
        ),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────────────────────

def load_pipeline(model_path: Optional[str], config_overrides: dict, device: torch.device):
    """Load evaluation pipeline (CausalDiT-based for benchmarking).

    Uses the standalone CausalDiT + StreamVAE pipeline which doesn't require
    the pretrained Z-Image-Turbo model. This is the standard benchmark configuration.
    """
    from inference.realtime_pipeline import RealtimePipeline, PipelineConfig
    from models import CausalDiT, ActionEncoder, StreamVAE

    config = PipelineConfig(
        height=config_overrides.get("height", 128),
        width=config_overrides.get("width", 128),
        num_inference_steps=config_overrides.get("num_steps", 2),
        compile_model=False,  # Skip compile for benchmarking (compile has overhead)
        use_kv_cache=True,
        use_motion_control=False,  # Skip for clean benchmarks
        device=str(device),
    )

    # Create CausalDiT model
    dit = CausalDiT(
        in_channels=16,
        hidden_dim=512,   # Small for benchmarking; real model uses 4096
        num_heads=8,
        num_layers=12,
        action_injection_layers=[4, 8],
    )

    action_encoder = ActionEncoder(
        num_actions=17,
        embedding_dim=128,
        hidden_dim=512,
    )

    # Stub VAE that works without actual pretrained weights
    # Maps (B, 3, H, W) → (B, 16, H//8, W//8)
    class StubVAE(nn.Module):
        def __init__(self, latent_channels=16, scale=8):
            super().__init__()
            self.latent_channels = latent_channels
            self.scale = scale
            self.encode_proj = nn.Conv2d(3, latent_channels * 2, scale, stride=scale)
            self.decode_proj = nn.ConvTranspose2d(latent_channels, 3, scale, stride=scale)

        def encode(self, x):
            # Ensure dtype matches model weights
            h = self.encode_proj(x.to(self.encode_proj.weight.dtype))
            mean, _ = h.chunk(2, dim=1)
            return mean

        def decode(self, z):
            return torch.tanh(self.decode_proj(z.to(self.decode_proj.weight.dtype)))

    dtype = getattr(torch, config.dtype)
    vae_nn = StubVAE().to(device=device, dtype=dtype)
    vae = StreamVAE(vae=vae_nn, tile_size=256)

    # Load weights if checkpoint provided
    if model_path and Path(model_path).exists():
        print(f"  Loading checkpoint: {model_path}")
        ckpt = torch.load(model_path, map_location=device)
        if "model_state_dict" in ckpt:
            dit.load_state_dict(ckpt["model_state_dict"], strict=False)
        if "action_encoder_state_dict" in ckpt:
            action_encoder.load_state_dict(ckpt["action_encoder_state_dict"], strict=False)

    pipeline = RealtimePipeline(dit, action_encoder, vae, config)
    return pipeline


def print_metrics(metrics: dict, title: str = ""):
    """Pretty-print evaluation metrics."""
    if title:
        print(f"\n{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:<40} {v:.4f}")
        else:
            print(f"  {k:<40} {v}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Z-Image World Model")
    parser.add_argument("--mode", choices=["fps", "quality", "action", "full"], default="fps",
                        help="Evaluation mode")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to model checkpoint")
    parser.add_argument("--video_path", type=str, default=None,
                        help="Path to reference video for quality evaluation")
    parser.add_argument("--num_frames", type=int, default=100,
                        help="Number of frames for FPS benchmark")
    parser.add_argument("--height", type=int, default=128,
                        help="Generation height")
    parser.add_argument("--width", type=int, default=128,
                        help="Generation width")
    parser.add_argument("--num_steps", type=int, default=2,
                        help="Number of denoising/flow steps")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"\nZ-Image World Model Evaluation")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"VRAM: {mem_gb:.1f} GB")

    # Load pipeline
    print("\nLoading pipeline...")
    pipeline = load_pipeline(
        args.model_path,
        {"num_steps": args.num_steps, "height": args.height, "width": args.width},
        device,
    )

    all_results = {}

    # FPS benchmark
    if args.mode in ("fps", "full"):
        print("\nRunning FPS benchmark...")
        fps_results = benchmark_fps(pipeline, num_frames=args.num_frames, device=device)
        all_results.update(fps_results)
        print_metrics(fps_results, "FPS Benchmark")

    # Quality evaluation (requires reference video)
    if args.mode in ("quality", "full"):
        if args.video_path:
            print("\nRunning quality evaluation...")
            quality_results = evaluate_generation_quality(
                pipeline, args.video_path, device=device
            )
            all_results.update(quality_results)
            print_metrics(quality_results, "Generation Quality")
        else:
            print("\nSkipping quality evaluation (no --video_path provided)")

    # Action responsiveness
    if args.mode in ("action", "full"):
        print("\nRunning action responsiveness evaluation...")
        initial_frame = torch.rand(1, 3, args.height, args.width, device=device)
        action_results = compute_action_responsiveness(
            pipeline, initial_frame, num_frames=16, device=device
        )
        all_results.update(action_results)
        print_metrics(action_results, "Action Responsiveness")

    # Summary
    print_metrics(all_results, "SUMMARY")

    return all_results


if __name__ == "__main__":
    main()
