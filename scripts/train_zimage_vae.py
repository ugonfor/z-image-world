#!/usr/bin/env python3
"""
Train next-frame predictor using Z-Image-Turbo's pretrained VAE.

Z-Image VAE: 38+ dB PSNR, 16-channel latents at H/8 × W/8.
This should produce much sharper results than our SimpleVAE (32 dB).

Uses multi-step rollout training to prevent autoregressive collapse.
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


# ─── Predictor (same U-Net architecture, adapted for 16 channels) ──

class DirectPredictor(nn.Module):
    """U-Net: predicts next_latent = current_latent + residual."""

    def __init__(self, channels=16, dim=128):
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
            nn.Conv2d(cin, cout, 3, 1, 1), nn.GroupNorm(min(8, cout), cout), nn.SiLU(),
            nn.Conv2d(cout, cout, 3, 1, 1), nn.GroupNorm(min(8, cout), cout), nn.SiLU(),
        )

    def forward(self, x):
        h1 = self.enc1(x)
        h2 = self.enc2(self.down1(h1))
        h = self.mid2(self.mid(self.down2(h2)))
        h = self.dec2(torch.cat([self.up1(h), h2], 1))
        h = self.dec1(torch.cat([self.up2(h), h1], 1))
        return x + self.out(h)


# ─── Z-Image VAE wrapper ─────────────────────────────────────────

class ZImageVAE:
    """Wrapper around Z-Image-Turbo's pretrained VAE."""

    def __init__(self, device="cuda"):
        from diffusers import AutoencoderKL

        print("Loading Z-Image-Turbo VAE...")
        self.vae = AutoencoderKL.from_pretrained(
            "Tongyi-MAI/Z-Image-Turbo",
            subfolder="vae",
            torch_dtype=torch.float32,
        ).to(device)
        self.vae.eval()
        self.vae.requires_grad_(False)

        self.scaling_factor = self.vae.config.scaling_factor  # 0.3611
        self.shift_factor = self.vae.config.shift_factor      # 0.1159
        self.device = device
        print(f"  scaling_factor={self.scaling_factor}, shift_factor={self.shift_factor}")

    @torch.no_grad()
    def encode(self, images):
        """Encode images (B, 3, H, W) in [0,1] → latents (B, 16, H/8, W/8)."""
        x = images * 2.0 - 1.0  # [0,1] → [-1,1]
        posterior = self.vae.encode(x)
        latents = posterior.latent_dist.sample()
        latents = (latents - self.shift_factor) * self.scaling_factor
        return latents

    @torch.no_grad()
    def decode(self, latents):
        """Decode latents (B, 16, H/8, W/8) → images (B, 3, H, W) in [0,1]."""
        z = latents / self.scaling_factor + self.shift_factor
        images = self.vae.decode(z).sample
        images = (images + 1.0) / 2.0
        return images.clamp(0, 1)


# ─── Data ─────────────────────────────────────────────────────────

def generate_video_frames(vid_idx, num_frames, resolution):
    """Generate one synthetic video."""
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

    return np.stack(frames)


def precompute_latents(vae, num_videos, num_frames, resolution, device, batch_encode=4):
    """Encode all videos to Z-Image latent space. Returns (V, F, 16, H/8, W/8)."""
    print(f"Precomputing Z-Image latents: {num_videos} videos × {num_frames} frames...")

    all_sequences = []
    for vid_idx in range(num_videos):
        frames = generate_video_frames(vid_idx, num_frames, resolution)
        frames_t = torch.from_numpy(frames).to(device)

        # Encode in small batches to avoid OOM
        latents = []
        for i in range(0, len(frames_t), batch_encode):
            z = vae.encode(frames_t[i:i+batch_encode])
            latents.append(z)
        latents = torch.cat(latents, dim=0)

        all_sequences.append(latents)
        if (vid_idx + 1) % 10 == 0:
            print(f"  {vid_idx+1}/{num_videos} encoded")

    data = torch.stack(all_sequences)  # (V, F, 16, H/8, W/8)
    mem_mb = data.element_size() * data.nelement() / 1e6
    print(f"Data: {data.shape}, {mem_mb:.1f} MB GPU")
    return data


# ─── Training ─────────────────────────────────────────────────────

def train_predictor(predictor, data, epochs=500, batch_size=128, lr=3e-4,
                    max_rollout=8, warmup_epochs=100):
    """Train with progressive multi-step rollouts."""
    V, NF, C, H, W = data.shape
    device = data.device

    print(f"\n=== Training ({epochs} epochs, warmup={warmup_epochs}, "
          f"max_rollout={max_rollout}, batch={batch_size}) ===")

    optimizer = torch.optim.AdamW(predictor.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr*0.01)

    predictor.train()
    t0 = time.time()
    best_loss = float('inf')

    for epoch in range(epochs):
        if epoch < warmup_epochs:
            R = 1
        else:
            progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
            R = min(max_rollout, 1 + int(progress * (max_rollout - 1)))

        max_start = NF - R
        n_batches = max(1, (V * max_start) // batch_size)
        total_loss, batches = 0, 0

        for _ in range(n_batches):
            vid_idx = torch.randint(0, V, (batch_size,), device=device)
            start_idx = torch.randint(0, max(1, max_start), (batch_size,), device=device)

            current = data[vid_idx, start_idx]
            loss = torch.tensor(0.0, device=device)

            for step in range(R):
                frame_idx = (start_idx + step + 1).clamp(max=NF - 1)
                target = data[vid_idx, frame_idx]

                predicted = predictor(current)
                weight = 1.0 / (1 + step * 0.3)
                loss = loss + weight * F.mse_loss(predicted, target)
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
    print(f"Done in {elapsed:.1f}s ({epochs/elapsed:.1f} epochs/s)")
    return predictor


# ─── Evaluation ───────────────────────────────────────────────────

@torch.no_grad()
def evaluate(predictor, vae, data, num_gen=15, output_dir="inference_output/zimage_vae"):
    """Generate and evaluate frames."""
    from PIL import Image

    predictor.eval()
    V, NF, C, H, W = data.shape

    for vid_idx in [0, 5, 10, 42]:
        if vid_idx >= V:
            continue

        out_dir = Path(output_dir) / f"vid_{vid_idx}"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Autoregressive generation
        current = data[vid_idx, 0:1]
        gen_latents = [current]
        for _ in range(min(num_gen, NF - 1)):
            current = predictor(current)
            gen_latents.append(current)

        gen_latents = torch.cat(gen_latents, dim=0)

        # Decode
        gen_frames = []
        for i in range(0, len(gen_latents), 4):
            f = vae.decode(gen_latents[i:i+4])
            gen_frames.append(f.cpu())
        gen_frames = torch.cat(gen_frames, dim=0)

        gt_latents = data[vid_idx, :gen_latents.shape[0]]
        gt_frames = []
        for i in range(0, len(gt_latents), 4):
            f = vae.decode(gt_latents[i:i+4])
            gt_frames.append(f.cpu())
        gt_frames = torch.cat(gt_frames, dim=0)

        # Save
        gen_pil, gt_pil = [], []
        for i in range(gen_frames.shape[0]):
            img = (gen_frames[i].permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
            Image.fromarray(img).save(out_dir / f"gen_{i:03d}.png")
            gen_pil.append(Image.fromarray(img))

            if i < gt_frames.shape[0]:
                img_gt = (gt_frames[i].permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
                Image.fromarray(img_gt).save(out_dir / f"gt_{i:03d}.png")
                gt_pil.append(Image.fromarray(img_gt))

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

        print(f"  Vid {vid_idx}: PSNR = " +
              " → ".join(f"{p:.1f}" for p in psnrs[:8]) +
              ("..." if len(psnrs) > 8 else ""))

    predictor.train()


# ─── Main ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_videos", type=int, default=100)
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_rollout", type=int, default=8)
    parser.add_argument("--warmup_epochs", type=int, default=100)
    parser.add_argument("--generate_frames", type=int, default=15)
    parser.add_argument("--dim", type=int, default=128)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    t_total = time.time()

    # Load Z-Image VAE
    vae = ZImageVAE(device)

    # Precompute latents
    data = precompute_latents(vae, args.num_videos, args.num_frames, args.resolution, device)

    # Create predictor
    predictor = DirectPredictor(channels=16, dim=args.dim).to(device)
    n_params = sum(p.numel() for p in predictor.parameters())
    print(f"Predictor: {n_params:,} params")

    # Evaluate baseline (identity)
    print("\n=== Before training ===")
    evaluate(predictor, vae, data, args.generate_frames, "inference_output/zimage_before")

    # Train
    predictor = train_predictor(
        predictor, data,
        epochs=args.epochs, batch_size=args.batch_size, lr=3e-4,
        max_rollout=args.max_rollout, warmup_epochs=args.warmup_epochs,
    )

    # Save
    ckpt_dir = Path("checkpoints/zimage_predictor")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(predictor.state_dict(), ckpt_dir / "predictor.pt")

    # Evaluate
    print("\n=== After training ===")
    evaluate(predictor, vae, data, args.generate_frames, "inference_output/zimage_after")

    elapsed = time.time() - t_total
    print(f"\nALL DONE in {elapsed:.1f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
