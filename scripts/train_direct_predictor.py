#!/usr/bin/env python3
"""
Direct Next-Frame Predictor (No Diffusion)

Simplest possible world model: predict next frame latent from current frame latent.
No noise schedule, no denoising loop. Just direct regression.

This serves as the PoC baseline - if this works, we know the architecture is sound
and can add diffusion for better quality later.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from einops import rearrange

from scripts.train_world_model import (
    SimpleVAE,
    SimpleVideoDataset,
    download_sample_videos,
    generate_synthetic_videos,
    save_frames_as_images,
    save_frames_as_video,
)


class DirectPredictor(nn.Module):
    """Predicts next frame latent from current frame latent directly.

    Uses a U-Net-like architecture in latent space.
    Input: current frame latent (B, C, H, W)
    Output: predicted next frame latent (B, C, H, W)
    """

    def __init__(self, latent_channels: int = 4, base_dim: int = 128):
        super().__init__()

        self.latent_channels = latent_channels

        # Encoder path (32x32 → 16x16 → 8x8)
        self.enc1 = nn.Sequential(
            nn.Conv2d(latent_channels, base_dim, 3, padding=1),
            nn.GroupNorm(8, base_dim),
            nn.SiLU(),
            nn.Conv2d(base_dim, base_dim, 3, padding=1),
            nn.GroupNorm(8, base_dim),
            nn.SiLU(),
        )
        self.down1 = nn.Conv2d(base_dim, base_dim, 4, 2, 1)  # 32→16

        self.enc2 = nn.Sequential(
            nn.Conv2d(base_dim, base_dim * 2, 3, padding=1),
            nn.GroupNorm(8, base_dim * 2),
            nn.SiLU(),
            nn.Conv2d(base_dim * 2, base_dim * 2, 3, padding=1),
            nn.GroupNorm(8, base_dim * 2),
            nn.SiLU(),
        )
        self.down2 = nn.Conv2d(base_dim * 2, base_dim * 2, 4, 2, 1)  # 16→8

        # Bottleneck (8x8)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_dim * 2, base_dim * 4, 3, padding=1),
            nn.GroupNorm(8, base_dim * 4),
            nn.SiLU(),
            nn.Conv2d(base_dim * 4, base_dim * 4, 3, padding=1),
            nn.GroupNorm(8, base_dim * 4),
            nn.SiLU(),
            nn.Conv2d(base_dim * 4, base_dim * 2, 3, padding=1),
            nn.GroupNorm(8, base_dim * 2),
            nn.SiLU(),
        )

        # Decoder path (8→16→32)
        self.up1 = nn.ConvTranspose2d(base_dim * 2, base_dim * 2, 4, 2, 1)
        self.dec2 = nn.Sequential(
            nn.Conv2d(base_dim * 4, base_dim * 2, 3, padding=1),  # skip connection
            nn.GroupNorm(8, base_dim * 2),
            nn.SiLU(),
            nn.Conv2d(base_dim * 2, base_dim, 3, padding=1),
            nn.GroupNorm(8, base_dim),
            nn.SiLU(),
        )

        self.up2 = nn.ConvTranspose2d(base_dim, base_dim, 4, 2, 1)
        self.dec1 = nn.Sequential(
            nn.Conv2d(base_dim * 2, base_dim, 3, padding=1),  # skip connection
            nn.GroupNorm(8, base_dim),
            nn.SiLU(),
            nn.Conv2d(base_dim, base_dim, 3, padding=1),
            nn.GroupNorm(8, base_dim),
            nn.SiLU(),
        )

        # Output: predict residual (next - current)
        self.out = nn.Conv2d(base_dim, latent_channels, 3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x):
        """Predict next frame latent as current + residual."""
        # Encoder
        h1 = self.enc1(x)
        h2 = self.enc2(self.down1(h1))
        h = self.bottleneck(self.down2(h2))

        # Decoder with skip connections
        h = self.up1(h)
        h = self.dec2(torch.cat([h, h2], dim=1))
        h = self.up2(h)
        h = self.dec1(torch.cat([h, h1], dim=1))

        # Predict residual and add to input
        residual = self.out(h)
        return x + residual


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Get videos
    data_dir = Path("data/videos")
    video_paths = download_sample_videos(data_dir, args.num_videos)

    if len(video_paths) < args.num_videos:
        video_paths.extend(
            generate_synthetic_videos(data_dir, args.num_videos - len(video_paths), start_idx=len(video_paths))
        )

    dataset = SimpleVideoDataset(
        video_paths=video_paths,
        num_frames=args.num_frames,
        resolution=(args.resolution, args.resolution),
        samples_per_video=args.samples_per_video,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    print(f"Dataset: {len(dataset)} samples, {len(dataloader)} batches")

    # Load VAE (use pretrained if available)
    vae = SimpleVAE(latent_channels=4).to(device)
    vae_path = Path("checkpoints/world_model/vae.pt")
    if vae_path.exists():
        print(f"Loading pretrained VAE from {vae_path}")
        vae.load_state_dict(torch.load(vae_path, map_location=device))
    else:
        print("No pretrained VAE found, training from scratch...")
        from scripts.train_world_model import train_vae
        vae = train_vae(vae, dataloader, num_epochs=30, device=device)
        vae_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(vae.state_dict(), vae_path)

    vae.eval()

    # Create predictor
    predictor = DirectPredictor(latent_channels=4, base_dim=args.base_dim).to(device)
    num_params = sum(p.numel() for p in predictor.parameters())
    print(f"Predictor parameters: {num_params:,}")

    optimizer = torch.optim.AdamW(predictor.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    # Training
    print(f"\n=== Training Direct Predictor ({args.epochs} epochs) ===")
    predictor.train()

    for epoch in range(args.epochs):
        total_loss = 0
        num_batches = 0

        for batch in dataloader:
            frames = batch["frames"].to(device)
            batch_size, num_frames = frames.shape[:2]

            # Encode all frames
            with torch.no_grad():
                frames_flat = rearrange(frames, "b f c h w -> (b f) c h w")
                latents, _, _ = vae.encode(frames_flat)
                latents = rearrange(latents, "(b f) c h w -> b f c h w", b=batch_size, f=num_frames)

            # For each frame pair: predict next from current
            losses = []
            for f in range(num_frames - 1):
                current = latents[:, f]
                target = latents[:, f + 1]

                predicted = predictor(current)
                loss = F.mse_loss(predicted, target)
                losses.append(loss)

            loss = torch.stack(losses).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        scheduler.step()
        avg_loss = total_loss / num_batches
        lr_now = scheduler.get_last_lr()[0]
        print(f"  Epoch {epoch+1}/{args.epochs}: loss={avg_loss:.6f}, lr={lr_now:.2e}")

    # Save
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = output_dir / "direct_predictor.pt"
    torch.save({
        "predictor_state_dict": predictor.state_dict(),
        "config": {"resolution": args.resolution, "latent_channels": 4, "base_dim": args.base_dim},
    }, ckpt_path)
    print(f"\nSaved predictor to {ckpt_path}")

    # Generate test frames
    print("\n=== Testing Inference ===")
    predictor.eval()

    # Get initial frame
    sample = dataset[0]
    initial_frame = sample["frames"][0].to(device)  # (3, H, W)

    # Encode
    with torch.no_grad():
        initial_latent, _, _ = vae.encode(initial_frame.unsqueeze(0))  # (1, 4, h, w)

    generated_frames = [initial_frame.unsqueeze(0).cpu()]
    current_latent = initial_latent

    for f in range(args.generate_frames):
        with torch.no_grad():
            next_latent = predictor(current_latent)
            frame = vae.decode(next_latent)
        generated_frames.append(frame.cpu())
        current_latent = next_latent

        if (f + 1) % 5 == 0:
            print(f"  Generated frame {f+1}/{args.generate_frames}")

    # Save outputs
    inference_dir = output_dir / "inference_output"
    save_frames_as_images(generated_frames, inference_dir)
    save_frames_as_video(generated_frames, output_dir / "generated.mp4")

    # Ground truth
    gt_frames = [sample["frames"][i].unsqueeze(0) for i in range(min(len(sample["frames"]), args.generate_frames + 1))]
    save_frames_as_images(gt_frames, output_dir / "ground_truth")

    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Generated frames: {inference_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_videos", type=int, default=50)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--num_frames", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--samples_per_video", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--base_dim", type=int, default=128)
    parser.add_argument("--generate_frames", type=int, default=30)
    parser.add_argument("--output_dir", type=str, default="checkpoints/direct_predictor")
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
