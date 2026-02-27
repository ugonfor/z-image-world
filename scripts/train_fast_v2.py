#!/usr/bin/env python3
"""
Fast Training v2 — Fixes autoregressive collapse with multi-step rollout training.

v1 problem: trained on single-step (frame[t]→frame[t+1]) but collapsed
during multi-step generation because errors compound.

v2 fix: Train with multi-step rollouts so the model learns to correct its own errors.
All data stored as a single GPU tensor for maximum speed.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import numpy as np


# ─── Models ──────────────────────────────────────────────────────────

class DirectPredictor(nn.Module):
    """U-Net in latent space: predicts next_latent = current_latent + residual."""

    def __init__(self, channels=4, dim=128):
        super().__init__()
        self.enc1 = self._block(channels, dim)
        self.down1 = nn.Conv2d(dim, dim, 4, 2, 1)
        self.enc2 = self._block(dim, dim * 2)
        self.down2 = nn.Conv2d(dim * 2, dim * 2, 4, 2, 1)
        self.mid = self._block(dim * 2, dim * 4)
        self.mid2 = self._block(dim * 4, dim * 2)
        self.up1 = nn.ConvTranspose2d(dim * 2, dim * 2, 4, 2, 1)
        self.dec2 = self._block(dim * 4, dim)
        self.up2 = nn.ConvTranspose2d(dim, dim, 4, 2, 1)
        self.dec1 = self._block(dim * 2, dim)
        self.out = nn.Conv2d(dim, channels, 3, 1, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def _block(self, cin, cout):
        return nn.Sequential(
            nn.Conv2d(cin, cout, 3, 1, 1), nn.GroupNorm(8, cout), nn.SiLU(),
            nn.Conv2d(cout, cout, 3, 1, 1), nn.GroupNorm(8, cout), nn.SiLU(),
        )

    def forward(self, x):
        h1 = self.enc1(x)
        h2 = self.enc2(self.down1(h1))
        h = self.mid2(self.mid(self.down2(h2)))
        h = self.dec2(torch.cat([self.up1(h), h2], 1))
        h = self.dec1(torch.cat([self.up2(h), h1], 1))
        return x + self.out(h)


class SimpleVAE(nn.Module):
    def __init__(self, latent_channels=4, base=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, base, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(base, base * 2, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(base * 2, base * 4, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(base * 4, latent_channels * 2, 3, 1, 1),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_channels, base * 4, 3, 1, 1), nn.ReLU(),
            nn.ConvTranspose2d(base * 4, base * 2, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(base * 2, base, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(base, 3, 4, 2, 1), nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.encoder(x)
        mu, logvar = h.chunk(2, dim=1)
        return mu + torch.randn_like(mu) * (0.5 * logvar).exp(), mu, logvar

    def decode(self, z):
        return self.decoder(z)


# ─── Data: single tensor on GPU ──────────────────────────────────

def generate_video_frames(vid_idx, num_frames, resolution):
    """Generate one synthetic video's frames as numpy array."""
    np.random.seed(vid_idx)
    motion = np.random.choice(['circle', 'linear', 'zoom', 'rotate'])
    color = np.random.rand(3).astype(np.float32)
    h, w = resolution, resolution

    frames = []
    for f_idx in range(num_frames):
        t = f_idx / num_frames
        y_grid, x_grid = np.mgrid[0:h, 0:w].astype(np.float32)
        x_norm, y_norm = x_grid / w, y_grid / h

        if motion == 'circle':
            cx, cy = w/2 + w/4 * np.cos(2*np.pi*t), h/2 + h/4 * np.sin(2*np.pi*t)
            pattern = np.exp(-((x_grid-cx)**2 + (y_grid-cy)**2) / (w/4)**2)
        elif motion == 'linear':
            pattern = np.sin(2*np.pi*(x_norm+t)*3) * 0.5 + 0.5
        elif motion == 'zoom':
            scale = 1 + t*2
            dist = np.sqrt((x_grid-w/2)**2 + (y_grid-h/2)**2)
            pattern = np.sin(dist*scale*0.1) * 0.5 + 0.5
        else:
            angle = t * 2 * np.pi
            xr = (x_norm-0.5)*np.cos(angle) - (y_norm-0.5)*np.sin(angle)
            yr = (x_norm-0.5)*np.sin(angle) + (y_norm-0.5)*np.cos(angle)
            pattern = (np.sin(xr*10)*np.sin(yr*10))*0.5 + 0.5

        frame = np.stack([pattern * c for c in color], axis=0)
        noise = np.random.rand(3, h, w).astype(np.float32) * 0.08 - 0.04
        frame = np.clip(frame + noise, 0, 1).astype(np.float32)
        frames.append(frame)

    return np.stack(frames)  # (F, 3, H, W)


def load_all_sequences(vae, num_videos, num_frames, resolution, device):
    """Encode all videos into a single (V, F, C, H, W) tensor on GPU."""
    print(f"Precomputing latents: {num_videos} videos × {num_frames} frames...")

    all_latents = []
    for vid_idx in range(num_videos):
        frames = generate_video_frames(vid_idx, num_frames, resolution)
        frames_t = torch.from_numpy(frames).to(device)

        with torch.no_grad():
            z, _, _ = vae.encode(frames_t)
        all_latents.append(z)

        if (vid_idx + 1) % 20 == 0:
            print(f"  {vid_idx+1}/{num_videos} encoded")

    # Stack into (V, F, C, H, W)
    data = torch.stack(all_latents)
    mem_mb = data.element_size() * data.nelement() / 1e6
    print(f"Data shape: {data.shape}, GPU memory: {mem_mb:.1f} MB")
    return data


# ─── Training ─────────────────────────────────────────────────────

def train_v2(predictor, data, epochs=1000, batch_size=128, lr=3e-4,
             max_rollout=8, warmup_epochs=200):
    """
    Train with progressive multi-step rollouts.

    data: (V, F, C, H, W) — all video latent sequences on GPU.

    Phase 1 (epoch 0..warmup): single-step (rollout=1)
    Phase 2 (epoch warmup..end): rollout gradually increases to max_rollout
    """
    V, NF, C, H, W = data.shape
    device = data.device
    print(f"\n=== Training v2 ({epochs} epochs, warmup={warmup_epochs}, "
          f"max_rollout={max_rollout}, batch={batch_size}) ===")

    optimizer = torch.optim.AdamW(predictor.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr*0.01)

    predictor.train()
    t0 = time.time()
    best_loss = float('inf')

    for epoch in range(epochs):
        # Determine rollout length
        if epoch < warmup_epochs:
            R = 1
        else:
            progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
            R = min(max_rollout, 1 + int(progress * (max_rollout - 1)))

        total_loss, batches = 0, 0
        max_start = NF - R  # can start from frame 0..NF-R-1

        # How many batches per epoch: ~(V * max_start) / batch_size
        n_batches = max(1, (V * max_start) // batch_size)

        for _ in range(n_batches):
            # Sample random (video, start_frame) pairs
            vid_idx = torch.randint(0, V, (batch_size,), device=device)
            start_idx = torch.randint(0, max(1, max_start), (batch_size,), device=device)

            # Get starting latents: (B, C, H, W)
            current = data[vid_idx, start_idx]

            loss = torch.tensor(0.0, device=device)

            for step in range(R):
                frame_idx = start_idx + step + 1
                # Clamp to valid range
                frame_idx = frame_idx.clamp(max=NF - 1)
                target = data[vid_idx, frame_idx]

                predicted = predictor(current)
                step_loss = F.mse_loss(predicted, target)

                # Weight: later steps count less
                weight = 1.0 / (1 + step * 0.3)
                loss = loss + weight * step_loss

                # Use predicted as input for next step (autoregressive)
                current = predicted

            loss = loss / R

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            batches += 1

        scheduler.step()
        avg_loss = total_loss / max(1, batches)
        if avg_loss < best_loss:
            best_loss = avg_loss

        if (epoch + 1) % 50 == 0 or epoch == 0 or epoch < 10:
            elapsed = time.time() - t0
            print(f"  Epoch {epoch+1}/{epochs}: loss={avg_loss:.6f}, best={best_loss:.6f}, "
                  f"R={R}, lr={scheduler.get_last_lr()[0]:.2e} [{elapsed:.1f}s]")

    elapsed = time.time() - t0
    print(f"Training done in {elapsed:.1f}s ({epochs/elapsed:.1f} epochs/s)")
    return predictor


# ─── Inference & Evaluation ───────────────────────────────────────

@torch.no_grad()
def generate_and_evaluate(predictor, vae, data, num_gen=15, output_dir="inference_output/v2"):
    """Generate frames from first frame of several videos and evaluate quality."""
    from PIL import Image

    predictor.eval()
    V, NF, C, H, W = data.shape

    for vid_idx in [0, 5, 10, 42]:
        if vid_idx >= V:
            continue

        out_dir = Path(output_dir) / f"vid_{vid_idx}"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Generate autoregressively
        current = data[vid_idx, 0:1]  # (1, C, H, W)
        gen_latents = [current]
        for _ in range(min(num_gen, NF - 1)):
            current = predictor(current)
            gen_latents.append(current)

        gen_latents = torch.cat(gen_latents, dim=0)  # (N, C, H, W)
        gen_frames = vae.decode(gen_latents)

        # Ground truth
        gt_latents = data[vid_idx, :gen_latents.shape[0]]
        gt_frames = vae.decode(gt_latents)

        # Save
        gen_pil, gt_pil = [], []
        for i in range(gen_frames.shape[0]):
            img = (gen_frames[i].cpu().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
            Image.fromarray(img).save(out_dir / f"gen_{i:03d}.png")
            gen_pil.append(Image.fromarray(img))

            if i < gt_frames.shape[0]:
                img_gt = (gt_frames[i].cpu().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
                Image.fromarray(img_gt).save(out_dir / f"gt_{i:03d}.png")
                gt_pil.append(Image.fromarray(img_gt))

        # GIF
        if gen_pil:
            gen_pil[0].save(out_dir / "gen.gif", save_all=True, append_images=gen_pil[1:],
                           duration=150, loop=0)
        if gt_pil:
            gt_pil[0].save(out_dir / "gt.gif", save_all=True, append_images=gt_pil[1:],
                           duration=150, loop=0)

        # PSNR
        n = min(gen_frames.shape[0], gt_frames.shape[0])
        psnrs = []
        for i in range(n):
            mse = F.mse_loss(gen_frames[i:i+1], gt_frames[i:i+1]).item()
            psnr = -10 * np.log10(mse + 1e-8)
            psnrs.append(psnr)

        print(f"  Video {vid_idx}: PSNR = " +
              " → ".join(f"{p:.1f}" for p in psnrs[:8]) +
              ("..." if len(psnrs) > 8 else ""))

    predictor.train()


# ─── Main ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_videos", type=int, default=100)
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_rollout", type=int, default=8)
    parser.add_argument("--warmup_epochs", type=int, default=200)
    parser.add_argument("--generate_frames", type=int, default=15)
    parser.add_argument("--from_scratch", action="store_true", help="Don't load v1 checkpoint")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    t_total = time.time()

    # Load VAE
    vae = SimpleVAE().to(device)
    ckpt = torch.load("checkpoints/world_model/vae.pt", map_location=device)
    vae.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
    vae.eval()
    print("Loaded pretrained VAE")

    # Precompute data
    data = load_all_sequences(vae, args.num_videos, args.num_frames, args.resolution, device)

    # Create predictor
    predictor = DirectPredictor(channels=4, dim=128).to(device)

    if not args.from_scratch:
        v1_ckpt = Path("checkpoints/fast/direct_predictor.pt")
        if v1_ckpt.exists():
            predictor.load_state_dict(torch.load(v1_ckpt, map_location=device))
            print("Loaded v1 checkpoint as starting point")

    n_params = sum(p.numel() for p in predictor.parameters())
    print(f"DirectPredictor: {n_params:,} params")

    # Evaluate before training
    print("\n=== Before v2 training ===")
    generate_and_evaluate(predictor, vae, data, args.generate_frames, "inference_output/v2_before")

    # Train
    predictor = train_v2(
        predictor, data,
        epochs=args.epochs, batch_size=args.batch_size, lr=3e-4,
        max_rollout=args.max_rollout, warmup_epochs=args.warmup_epochs,
    )

    # Save
    ckpt_dir = Path("checkpoints/fast_v2")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(predictor.state_dict(), ckpt_dir / "direct_predictor.pt")

    # Evaluate after training
    print("\n=== After v2 training ===")
    generate_and_evaluate(predictor, vae, data, args.generate_frames, "inference_output/v2_after")

    elapsed = time.time() - t_total
    print(f"\nALL DONE in {elapsed:.1f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
