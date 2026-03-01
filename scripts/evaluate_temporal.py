#!/usr/bin/env python3
"""
Temporal Coherence Evaluation for ZImageWorldModel

Loads a trained checkpoint, generates multi-frame sequences from
synthetic seed frames, and measures temporal consistency metrics.

Usage:
    # Evaluate latest checkpoint
    python scripts/evaluate_temporal.py

    # Evaluate specific checkpoint
    python scripts/evaluate_temporal.py --checkpoint checkpoints/zimage_stage1_v2/world_model_epoch50.pt

    # Evaluate all checkpoints and compare
    python scripts/evaluate_temporal.py --all_checkpoints --checkpoint_dir checkpoints/zimage_stage1_v2
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
from einops import rearrange


def compute_temporal_consistency(frames: torch.Tensor) -> dict:
    """Measure frame-to-frame differences as temporal consistency proxy.

    Args:
        frames: (F, 3, H, W) tensor in [-1, 1] or [0, 1]

    Returns:
        Dict with consistency metrics
    """
    # Normalize to [0, 1]
    if frames.min() < -0.1:
        frames = (frames + 1) / 2

    frames = frames.clamp(0, 1)

    diffs = []
    for i in range(len(frames) - 1):
        diff = F.mse_loss(frames[i], frames[i + 1]).item()
        diffs.append(diff)

    return {
        "mean_frame_diff": sum(diffs) / len(diffs) if diffs else 0.0,
        "max_frame_diff": max(diffs) if diffs else 0.0,
        "min_frame_diff": min(diffs) if diffs else 0.0,
    }


def evaluate_checkpoint(
    checkpoint_path: str,
    model_path: str = "weights/Z-Image-Turbo",
    num_sequences: int = 5,
    num_frames: int = 8,
    resolution: int = 256,
    temporal_every_n: int = 3,
    device: str = "cuda",
    num_inference_steps: int = 2,
) -> dict:
    """Evaluate temporal coherence of a checkpoint.

    Generates frame sequences under two conditions:
    1. Spatial-only model (temporal disabled, gamma zeroed)
    2. Full model with trained temporal layers

    Returns metrics for both conditions to measure improvement.
    """
    from models.zimage_world_model import ZImageWorldModel

    print(f"Loading model from {model_path}...")
    model = ZImageWorldModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        temporal_every_n=temporal_every_n,
        freeze_spatial=True,
        device=device,
    )
    model.eval()
    print(f"Model loaded. Loading checkpoint {checkpoint_path}...")

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.temporal_layers.load_state_dict(ckpt["temporal_state_dict"])
    print(f"Checkpoint loaded (epoch {ckpt.get('epoch', '?')})")

    # Setup simple DDPM-style denoising
    T = 1000
    betas = torch.linspace(0.0001, 0.02, T, device=device)
    alphas = 1 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    def denoise_step(x_noisy, t_idx, pred):
        """Single DDIM denoising step from v-prediction."""
        alpha_t = alphas_cumprod[t_idx]
        beta_t = 1 - alpha_t
        # v = sqrt(alpha) * eps - sqrt(1 - alpha) * x0
        # x0 = sqrt(alpha) * x_noisy - sqrt(1 - alpha) * v
        x0_pred = alpha_t.sqrt() * x_noisy - beta_t.sqrt() * pred
        return x0_pred.clamp(-1, 1)

    # Choose timesteps for denoising
    step_indices = torch.linspace(T - 1, 0, num_inference_steps + 1, dtype=torch.long, device=device)
    t_start = step_indices[0]
    t_values = step_indices[1:]

    all_results = []
    temporal_diffs_with = []
    temporal_diffs_without = []

    print(f"\nGenerating {num_sequences} sequences ({num_frames} frames each)...")

    for seq_idx in range(num_sequences):
        torch.manual_seed(seq_idx * 42)

        # Seed frame (random, but same for both conditions)
        # VAE has 16 latent channels (not 4)
        latent_ch = model.vae.config.latent_channels  # 16
        seed_frame = torch.randn(1, latent_ch, resolution // 8, resolution // 8,
                                 device=device, dtype=torch.bfloat16)

        # --- WITH temporal layers ---
        frames_with = [seed_frame.clone()]
        prev_latent = seed_frame.clone()

        for f in range(num_frames - 1):
            # Add noise at level t_start
            noise = torch.randn_like(prev_latent)
            alpha_t = alphas_cumprod[t_start]
            x_noisy = alpha_t.sqrt() * prev_latent + (1 - alpha_t).sqrt() * noise

            with torch.inference_mode():
                # Forward pass with temporal context
                x_seq = x_noisy.unsqueeze(1)  # (1, 1, 4, H/8, W/8)
                t = t_start.float().unsqueeze(0)
                v_pred = model(x_seq, t)
                if v_pred.dim() == 5:
                    v_pred = v_pred[:, 0]

            x0 = denoise_step(x_noisy, t_start, v_pred)
            frames_with.append(x0.clone())
            prev_latent = x0.clone()

        # --- WITHOUT temporal layers (zero gamma) ---
        with torch.no_grad():
            # Temporarily zero all gammas
            gamma_backup = {}
            for key, param in model.temporal_layers.named_parameters():
                if "gamma" in key:
                    gamma_backup[key] = param.data.clone()
                    param.data.zero_()

        frames_without = [seed_frame.clone()]
        prev_latent = seed_frame.clone()

        for f in range(num_frames - 1):
            noise = torch.randn_like(prev_latent)
            alpha_t = alphas_cumprod[t_start]
            x_noisy = alpha_t.sqrt() * prev_latent + (1 - alpha_t).sqrt() * noise

            with torch.inference_mode():
                x_seq = x_noisy.unsqueeze(1)
                t = t_start.float().unsqueeze(0)
                v_pred = model(x_seq, t)
                if v_pred.dim() == 5:
                    v_pred = v_pred[:, 0]

            x0 = denoise_step(x_noisy, t_start, v_pred)
            frames_without.append(x0.clone())
            prev_latent = x0.clone()

        # Restore gammas
        with torch.no_grad():
            for key, param in model.temporal_layers.named_parameters():
                if key in gamma_backup:
                    param.data.copy_(gamma_backup[key])

        # Decode latents to pixel space for metric computation
        frames_with_px = []
        frames_without_px = []
        with torch.inference_mode():
            for lat in frames_with:
                px = model.vae.decode(lat / model.vae.config.scaling_factor).sample
                frames_with_px.append(px.squeeze(0))
            for lat in frames_without:
                px = model.vae.decode(lat / model.vae.config.scaling_factor).sample
                frames_without_px.append(px.squeeze(0))

        frames_with_stack = torch.stack(frames_with_px)
        frames_without_stack = torch.stack(frames_without_px)

        metrics_with = compute_temporal_consistency(frames_with_stack)
        metrics_without = compute_temporal_consistency(frames_without_stack)

        temporal_diffs_with.append(metrics_with["mean_frame_diff"])
        temporal_diffs_without.append(metrics_without["mean_frame_diff"])

        print(f"  Seq {seq_idx+1}: with_temporal={metrics_with['mean_frame_diff']:.4f}, "
              f"without={metrics_without['mean_frame_diff']:.4f}")

    mean_with = sum(temporal_diffs_with) / len(temporal_diffs_with)
    mean_without = sum(temporal_diffs_without) / len(temporal_diffs_without)

    improvement = (mean_without - mean_with) / mean_without * 100 if mean_without > 0 else 0

    results = {
        "checkpoint": str(checkpoint_path),
        "epoch": ckpt.get("epoch", "?"),
        "mean_temporal_diff_with_temporal": mean_with,
        "mean_temporal_diff_without_temporal": mean_without,
        "temporal_improvement_pct": improvement,
        "interpretation": "Lower frame diff = more temporally coherent/smooth",
    }

    print(f"\n=== Summary ===")
    print(f"  With temporal layers:    {mean_with:.4f} MSE")
    print(f"  Without temporal layers: {mean_without:.4f} MSE")
    print(f"  Temporal improvement:    {improvement:.1f}%")
    if improvement > 0:
        print(f"  ✓ Temporal layers improve coherence by {improvement:.1f}%")
    else:
        print(f"  ✗ Temporal layers did not improve coherence")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate temporal coherence")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint (default: auto-detect latest)")
    parser.add_argument("--checkpoint_dir", type=str,
                        default="checkpoints/zimage_stage1_v2",
                        help="Checkpoint directory")
    parser.add_argument("--model_path", type=str, default="weights/Z-Image-Turbo",
                        help="Base model path")
    parser.add_argument("--all_checkpoints", action="store_true",
                        help="Evaluate all checkpoints in directory")
    parser.add_argument("--num_sequences", type=int, default=5)
    parser.add_argument("--num_frames", type=int, default=8)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--temporal_every_n", type=int, default=3)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.all_checkpoints:
        ckpt_dir = Path(args.checkpoint_dir)
        checkpoints = sorted(ckpt_dir.glob("world_model_epoch*.pt"))
        if not checkpoints:
            print(f"No checkpoints found in {ckpt_dir}")
            return

        print(f"Evaluating {len(checkpoints)} checkpoints...")
        all_results = []
        for ckpt in checkpoints:
            print(f"\n{'='*60}")
            results = evaluate_checkpoint(
                str(ckpt),
                model_path=args.model_path,
                num_sequences=args.num_sequences,
                num_frames=args.num_frames,
                resolution=args.resolution,
                temporal_every_n=args.temporal_every_n,
                device=args.device,
            )
            all_results.append(results)

        print(f"\n{'='*60}")
        print("COMPARISON ACROSS EPOCHS:")
        print(f"{'Epoch':<10} {'With Temporal':<20} {'Without Temporal':<20} {'Improvement':<15}")
        print("-" * 65)
        for r in all_results:
            print(f"{r['epoch']:<10} {r['mean_temporal_diff_with_temporal']:<20.4f} "
                  f"{r['mean_temporal_diff_without_temporal']:<20.4f} "
                  f"{r['temporal_improvement_pct']:<15.1f}%")
    else:
        # Single checkpoint evaluation
        if args.checkpoint:
            ckpt_path = args.checkpoint
        else:
            ckpt_dir = Path(args.checkpoint_dir)
            ckpts = sorted(ckpt_dir.glob("world_model_epoch*.pt"))
            if not ckpts:
                # Try final
                final = ckpt_dir / "world_model_final.pt"
                if final.exists():
                    ckpt_path = str(final)
                else:
                    print("No checkpoints found.")
                    return
            else:
                ckpt_path = str(ckpts[-1])

        results = evaluate_checkpoint(
            ckpt_path,
            model_path=args.model_path,
            num_sequences=args.num_sequences,
            num_frames=args.num_frames,
            resolution=args.resolution,
            temporal_every_n=args.temporal_every_n,
            device=args.device,
        )


if __name__ == "__main__":
    main()
