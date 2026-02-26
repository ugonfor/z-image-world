#!/usr/bin/env python3
"""
Train Z-Image World Model

Loads the pretrained Z-Image-Turbo model and trains temporal attention
layers using Diffusion Forcing on video data.

Stage 1: Train temporal layers only (no actions) for multi-frame coherence
Stage 2: Add action conditioning (future work)

Usage:
    # Quick test with synthetic data
    uv run python scripts/train_zimage_world.py --quick

    # Train on video directory
    uv run python scripts/train_zimage_world.py --data_dir data/videos --epochs 10

    # With reduced temporal layers for faster training
    uv run python scripts/train_zimage_world.py --temporal_every_n 3 --epochs 20
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Force unbuffered output
sys.stdout = os.fdopen(sys.stdout.fileno(), "w", buffering=1)
sys.stderr = os.fdopen(sys.stderr.fileno(), "w", buffering=1)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from einops import rearrange

sys.path.insert(0, str(Path(__file__).parent.parent))

from training.diffusion_forcing import DiffusionForcingConfig, DiffusionForcingLoss


class SyntheticVideoDataset(Dataset):
    """Synthetic video dataset for testing the training pipeline."""

    def __init__(self, num_samples: int = 100, num_frames: int = 4, resolution: int = 256):
        self.num_samples = num_samples
        self.num_frames = num_frames
        self.resolution = resolution

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate synthetic frames: moving colored rectangles
        torch.manual_seed(idx)
        frames = torch.rand(self.num_frames, 3, self.resolution, self.resolution)
        # Add temporal coherence: each frame is a slight modification of the previous
        for f in range(1, self.num_frames):
            frames[f] = frames[f - 1] * 0.8 + frames[f] * 0.2
        return {"frames": frames}


class VideoFolderDataset(Dataset):
    """Dataset loading video files from a directory."""

    def __init__(self, data_dir: str, num_frames: int = 4, resolution: int = 256):
        self.data_dir = Path(data_dir)
        self.num_frames = num_frames
        self.resolution = resolution
        self.video_paths = sorted(
            p for p in self.data_dir.iterdir()
            if p.suffix.lower() in {".mp4", ".avi", ".mov", ".webm"}
        )
        if not self.video_paths:
            raise ValueError(f"No video files found in {data_dir}")
        print(f"Found {len(self.video_paths)} videos in {data_dir}")

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        import cv2
        cap = cv2.VideoCapture(str(self.video_paths[idx]))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Sample random contiguous frames
        if total_frames <= self.num_frames:
            start = 0
        else:
            start = torch.randint(0, total_frames - self.num_frames, (1,)).item()

        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        frames = []
        for _ in range(self.num_frames):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (self.resolution, self.resolution))
            frames.append(torch.from_numpy(frame).float() / 255.0)
        cap.release()

        # Pad if we got fewer frames than needed
        while len(frames) < self.num_frames:
            frames.append(frames[-1].clone())

        # Stack: (num_frames, 3, H, W)
        frames = torch.stack(frames)
        frames = rearrange(frames, "f h w c -> f c h w")
        return {"frames": frames}


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info(0)
        print(f"GPU Memory: {free / 1e9:.1f}GB free / {total / 1e9:.1f}GB total")

    # --- Load model ---
    print("\n=== Loading Z-Image World Model ===")
    from models.zimage_world_model import ZImageWorldModel

    model = ZImageWorldModel.from_pretrained(
        "Tongyi-MAI/Z-Image-Turbo",
        torch_dtype=torch.bfloat16,
        temporal_every_n=args.temporal_every_n,
        freeze_spatial=True,
        device=device,
    )

    print(f"Trainable params: {model.num_trainable_params() / 1e6:.1f}M")

    # Enable gradient checkpointing on transformer
    if hasattr(model.transformer, "gradient_checkpointing_enable"):
        model.transformer.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled")

    # --- Dataset ---
    print("\n=== Loading Dataset ===")
    resolution = args.resolution
    num_frames = args.num_frames

    if args.data_dir and Path(args.data_dir).exists():
        dataset = VideoFolderDataset(args.data_dir, num_frames=num_frames, resolution=resolution)
    else:
        print("Using synthetic video dataset for testing")
        dataset = SyntheticVideoDataset(
            num_samples=args.num_samples, num_frames=num_frames, resolution=resolution,
        )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # Keep simple for now
        pin_memory=True,
    )
    print(f"Dataset: {len(dataset)} samples, batch_size={args.batch_size}")

    # --- Optimizer ---
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=0.01,
        betas=(0.9, 0.999),
    )

    # --- Loss function ---
    df_config = DiffusionForcingConfig(
        num_train_timesteps=1000,
        beta_schedule="scaled_linear",
        prediction_type="v_prediction",
        independent_noise=True,
        noise_level_sampling="pyramid",
        num_frames=num_frames,
    )
    loss_fn = DiffusionForcingLoss(df_config).to(device)

    # --- Checkpoint directory ---
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # --- Training loop ---
    print(f"\n=== Training ({args.epochs} epochs) ===")
    model.train()
    # Only temporal + action layers need training mode
    model.transformer.eval()  # Keep pretrained transformer in eval mode
    model.vae.eval()

    grad_accum = args.grad_accum
    global_step = 0

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        epoch_steps = 0
        t_start = time.time()

        for batch_idx, batch in enumerate(dataloader):
            frames = batch["frames"].to(device, dtype=torch.bfloat16)  # (B, F, 3, H, W)
            batch_size = frames.shape[0]

            # Encode frames to latents
            with torch.no_grad():
                latents = model.encode_frames(frames)  # (B, F, 16, H//8, W//8)

            # Sample timesteps and noise
            timesteps = loss_fn.sample_timesteps(batch_size, num_frames, device)
            noise = torch.randn_like(latents)
            noisy_latents = loss_fn.add_noise(latents, noise, timesteps)

            # Forward pass
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                # ZImageWorldModel.forward returns just the prediction (no cache tuple)
                model_output = model(noisy_latents, timesteps.float())

                # Ensure model_output matches target shape
                if model_output.dim() == 4 and latents.dim() == 5:
                    model_output = model_output.unsqueeze(1)

                loss_dict = loss_fn(model_output, latents, noise, timesteps)

            loss = loss_dict["loss"] / grad_accum
            loss.backward()

            epoch_loss += loss_dict["loss"].item()
            epoch_steps += 1

            # Gradient step
            if (batch_idx + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

        # End of epoch
        elapsed = time.time() - t_start
        avg_loss = epoch_loss / max(epoch_steps, 1)
        print(f"  Epoch {epoch + 1}/{args.epochs}: loss={avg_loss:.4f}, time={elapsed:.1f}s, steps={global_step}")

        # Save checkpoint periodically
        if (epoch + 1) % args.save_every == 0 or (epoch + 1) == args.epochs:
            ckpt_path = ckpt_dir / f"world_model_epoch{epoch + 1}.pt"
            torch.save({
                "epoch": epoch + 1,
                "global_step": global_step,
                "temporal_state_dict": model.temporal_layers.state_dict(),
                "action_injections_state_dict": model.action_injections.state_dict(),
                "action_encoder_state_dict": model.action_encoder.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": {
                    "temporal_every_n": args.temporal_every_n,
                    "num_frames": num_frames,
                    "resolution": resolution,
                },
            }, ckpt_path)
            print(f"  Saved checkpoint: {ckpt_path}")

    # Final checkpoint
    final_path = ckpt_dir / "world_model_final.pt"
    torch.save({
        "epoch": args.epochs,
        "global_step": global_step,
        "temporal_state_dict": model.temporal_layers.state_dict(),
        "action_injections_state_dict": model.action_injections.state_dict(),
        "action_encoder_state_dict": model.action_encoder.state_dict(),
        "config": {
            "temporal_every_n": args.temporal_every_n,
            "num_frames": num_frames,
            "resolution": resolution,
        },
    }, final_path)

    print(f"\n{'=' * 60}")
    print("TRAINING COMPLETE")
    print(f"{'=' * 60}")
    print(f"Final checkpoint: {final_path}")
    print(f"Total steps: {global_step}")

    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info(0)
        print(f"GPU Memory: {free / 1e9:.1f}GB free / {total / 1e9:.1f}GB total")


def main():
    parser = argparse.ArgumentParser(description="Train Z-Image World Model")
    parser.add_argument("--data_dir", type=str, default=None, help="Video directory")
    parser.add_argument("--resolution", type=int, default=256, help="Training resolution")
    parser.add_argument("--num_frames", type=int, default=4, help="Frames per sample")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--temporal_every_n", type=int, default=1, help="Temporal attention every N layers")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/zimage_world", help="Checkpoint directory")
    parser.add_argument("--save_every", type=int, default=5, help="Save checkpoint every N epochs")
    parser.add_argument("--num_samples", type=int, default=50, help="Number of synthetic samples")
    parser.add_argument("--quick", action="store_true", help="Quick test run")
    args = parser.parse_args()

    if args.quick:
        args.epochs = 2
        args.num_samples = 10
        args.batch_size = 1
        args.num_frames = 2
        args.resolution = 128
        args.grad_accum = 1

    train(args)


if __name__ == "__main__":
    main()
