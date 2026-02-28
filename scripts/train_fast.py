#!/usr/bin/env python3
"""
Fast Training Pipeline — Optimized for DGX Spark (128GB GPU)

Key optimizations:
1. Precompute ALL video frames → VAE latents in GPU memory (no disk I/O during training)
2. Large batch sizes (fill GPU memory)
3. Mixed precision (bf16)
4. No data loading overhead during training

Trains both:
- Direct predictor (simpler, faster convergence)
- Conditional diffusion (better quality)
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import argparse


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
        self.dec2 = self._block(dim * 4, dim)  # skip concat
        self.up2 = nn.ConvTranspose2d(dim, dim, 4, 2, 1)
        self.dec1 = self._block(dim * 2, dim)  # skip concat
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


class ConditionalDiT(nn.Module):
    """Conditional noise predictor: concat [noisy_target | cond] → predict noise."""

    def __init__(self, channels=4, dim=512, heads=8, layers=6, patch=4):
        super().__init__()
        self.channels, self.dim, self.patch = channels, dim, patch
        self.patch_embed = nn.Conv2d(channels * 2, dim, patch, patch)
        self.pos_embed = nn.Parameter(torch.zeros(1, 1024, dim))
        self.time_embed = nn.Sequential(nn.Linear(dim, dim * 4), nn.SiLU(), nn.Linear(dim * 4, dim))
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(dim, heads, dim * 4, 0.0, 'gelu', batch_first=True)
            for _ in range(layers)
        ])
        self.norm = nn.LayerNorm(dim)
        self.out_proj = nn.Linear(dim, patch * patch * channels)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def _t_emb(self, t, dim):
        half = dim // 2
        f = torch.exp(-torch.log(torch.tensor(10000.0)) * torch.arange(half, device=t.device) / half)
        e = t[:, None].float() * f[None, :]
        return torch.cat([e.sin(), e.cos()], -1)

    def forward(self, x, t, cond=None):
        B, C, H, W = x.shape
        if cond is None:
            cond = torch.zeros_like(x)
        x_in = self.patch_embed(torch.cat([x, cond], 1))
        h, w = x_in.shape[2:]
        x_in = rearrange(x_in, 'b d h w -> b (h w) d')
        x_in = x_in + self.pos_embed[:, :x_in.shape[1]]
        x_in = x_in + self.time_embed(self._t_emb(t, self.dim)).unsqueeze(1)
        for layer in self.layers:
            x_in = layer(x_in)
        x_in = self.out_proj(self.norm(x_in))
        return rearrange(x_in, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)',
                         h=h, w=w, p1=self.patch, p2=self.patch, c=C)


class SimpleVAE(nn.Module):
    """Same SimpleVAE as before."""

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

    def forward(self, x):
        z, mu, lv = self.encode(x)
        return self.decode(z), mu, lv


# ─── Data: precompute everything to GPU ──────────────────────────────

def load_all_video_latents(vae, num_videos=50, num_frames=8, resolution=256, device='cuda'):
    """Load ALL videos, encode to latents, store in GPU memory.

    Returns: (N_pairs, 2, C, H, W) tensor of (current, next) latent pairs on GPU.
    """
    import numpy as np

    print(f"Precomputing latents for {num_videos} videos × {num_frames} frames...")

    # Generate synthetic videos in memory (skip disk entirely for speed)
    all_pairs = []

    for vid_idx in range(num_videos):
        np.random.seed(vid_idx)
        motion = np.random.choice(['circle', 'linear', 'zoom', 'rotate'])
        color = np.random.rand(3).astype(np.float32)
        h, w = resolution, resolution

        frames_tensor = []
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

            frame = np.stack([pattern * c for c in color], axis=0)  # (3, H, W)
            noise = np.random.rand(3, h, w).astype(np.float32) * 0.08 - 0.04
            frame = np.clip(frame + noise, 0, 1).astype(np.float32)
            frames_tensor.append(torch.from_numpy(frame))

        # Stack and encode batch
        frames_batch = torch.stack(frames_tensor).to(device)  # (F, 3, H, W)

        with torch.no_grad():
            latents, _, _ = vae.encode(frames_batch)  # (F, 4, h, w)

        # Create frame pairs
        for f in range(num_frames - 1):
            pair = torch.stack([latents[f], latents[f + 1]])  # (2, 4, h, w)
            all_pairs.append(pair)

        if (vid_idx + 1) % 10 == 0:
            print(f"  {vid_idx+1}/{num_videos} videos encoded")

    all_pairs = torch.stack(all_pairs)  # (N, 2, 4, h, w)
    print(f"Precomputed {len(all_pairs)} frame pairs, shape={all_pairs.shape}")
    mem_mb = all_pairs.element_size() * all_pairs.nelement() / 1e6
    print(f"GPU memory for latents: {mem_mb:.1f} MB")

    return all_pairs


def load_all_video_frames(num_videos=50, num_frames=8, resolution=256, device='cuda'):
    """Load raw frames to GPU for VAE training."""
    import numpy as np

    print(f"Generating {num_videos} videos in memory...")
    all_frames = []

    for vid_idx in range(num_videos):
        np.random.seed(vid_idx)
        motion = np.random.choice(['circle', 'linear', 'zoom', 'rotate'])
        color = np.random.rand(3).astype(np.float32)
        h, w = resolution, resolution

        for f_idx in range(num_frames):
            t = f_idx / num_frames
            y_grid, x_grid = np.mgrid[0:h, 0:w].astype(np.float32)
            x_norm, y_norm = x_grid / w, y_grid / h

            if motion == 'circle':
                cx, cy = w/2 + w/4*np.cos(2*np.pi*t), h/2 + h/4*np.sin(2*np.pi*t)
                pattern = np.exp(-((x_grid-cx)**2 + (y_grid-cy)**2) / (w/4)**2)
            elif motion == 'linear':
                pattern = np.sin(2*np.pi*(x_norm+t)*3)*0.5 + 0.5
            elif motion == 'zoom':
                scale = 1+t*2
                dist = np.sqrt((x_grid-w/2)**2 + (y_grid-h/2)**2)
                pattern = np.sin(dist*scale*0.1)*0.5 + 0.5
            else:
                angle = t*2*np.pi
                xr = (x_norm-0.5)*np.cos(angle)-(y_norm-0.5)*np.sin(angle)
                yr = (x_norm-0.5)*np.sin(angle)+(y_norm-0.5)*np.cos(angle)
                pattern = (np.sin(xr*10)*np.sin(yr*10))*0.5+0.5

            frame = np.stack([pattern*c for c in color], axis=0)
            noise = np.random.rand(3, h, w).astype(np.float32)*0.08-0.04
            frame = np.clip(frame+noise, 0, 1).astype(np.float32)
            all_frames.append(torch.from_numpy(frame))

    all_frames = torch.stack(all_frames).to(device)  # (N*F, 3, H, W)
    mem_mb = all_frames.element_size() * all_frames.nelement() / 1e6
    print(f"All frames in GPU: {all_frames.shape}, {mem_mb:.1f} MB")
    return all_frames


# ─── Training loops ──────────────────────────────────────────────────

def train_vae_fast(vae, all_frames, epochs=30, batch_size=64, lr=1e-4):
    """Train VAE with all data pre-loaded in GPU."""
    print(f"\n=== Training VAE ({epochs} epochs, batch_size={batch_size}) ===")
    optimizer = torch.optim.AdamW(vae.parameters(), lr=lr)
    N = len(all_frames)
    vae.train()

    t0 = time.time()
    for epoch in range(epochs):
        perm = torch.randperm(N, device=all_frames.device)
        total_loss, total_recon, batches = 0, 0, 0

        for i in range(0, N, batch_size):
            idx = perm[i:i+batch_size]
            frames = all_frames[idx]

            recon, mu, lv = vae(frames)
            recon_loss = F.mse_loss(recon, frames)
            kl_loss = -0.5 * (1 + lv - mu.pow(2) - lv.exp()).mean()
            loss = recon_loss + 0.001 * kl_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_recon += recon_loss.item()
            batches += 1

        if (epoch + 1) % 5 == 0 or epoch == 0:
            elapsed = time.time() - t0
            print(f"  Epoch {epoch+1}/{epochs}: recon={total_recon/batches:.6f}, "
                  f"loss={total_loss/batches:.6f} [{elapsed:.1f}s]")

    elapsed = time.time() - t0
    print(f"VAE training done in {elapsed:.1f}s ({epochs/elapsed:.1f} epochs/s)")
    return vae


def train_direct_fast(predictor, pairs, epochs=500, batch_size=128, lr=3e-4):
    """Train direct predictor with all pairs in GPU."""
    print(f"\n=== Training Direct Predictor ({epochs} epochs, batch_size={batch_size}) ===")
    optimizer = torch.optim.AdamW(predictor.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr*0.01)
    N = len(pairs)
    predictor.train()

    t0 = time.time()
    best_loss = float('inf')
    for epoch in range(epochs):
        perm = torch.randperm(N, device=pairs.device)
        total_loss, batches = 0, 0

        for i in range(0, N, batch_size):
            idx = perm[i:i+batch_size]
            current = pairs[idx, 0]  # (B, C, H, W)
            target = pairs[idx, 1]

            predicted = predictor(current)
            loss = F.mse_loss(predicted, target)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            batches += 1

        scheduler.step()
        avg_loss = total_loss / batches

        if avg_loss < best_loss:
            best_loss = avg_loss

        if (epoch + 1) % 50 == 0 or epoch == 0 or epoch < 10:
            elapsed = time.time() - t0
            print(f"  Epoch {epoch+1}/{epochs}: loss={avg_loss:.6f}, best={best_loss:.6f}, "
                  f"lr={scheduler.get_last_lr()[0]:.2e} [{elapsed:.1f}s]")

    elapsed = time.time() - t0
    print(f"Direct predictor done in {elapsed:.1f}s ({epochs/elapsed:.1f} epochs/s)")
    return predictor


def train_diffusion_fast(dit, pairs, epochs=500, batch_size=128, lr=1e-4, num_timesteps=1000):
    """Train conditional diffusion with all pairs in GPU."""
    print(f"\n=== Training Conditional Diffusion ({epochs} epochs, batch_size={batch_size}) ===")
    device = pairs.device
    optimizer = torch.optim.AdamW(dit.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr*0.01)

    betas = torch.linspace(0.0001, 0.02, num_timesteps, device=device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    sqrt_ac = torch.sqrt(alphas_cumprod)
    sqrt_1mac = torch.sqrt(1 - alphas_cumprod)

    N = len(pairs)
    dit.train()

    t0 = time.time()
    best_loss = float('inf')
    for epoch in range(epochs):
        perm = torch.randperm(N, device=device)
        total_loss, batches = 0, 0

        for i in range(0, N, batch_size):
            idx = perm[i:i+batch_size]
            current = pairs[idx, 0]
            target = pairs[idx, 1]
            B = current.shape[0]

            # Noise augmentation on condition (10% chance)
            cond = current.clone()
            if torch.rand(1).item() < 0.1:
                cond = cond + 0.05 * torch.randn_like(cond)

            t = torch.randint(0, num_timesteps, (B,), device=device)
            noise = torch.randn_like(target)
            noisy = sqrt_ac[t, None, None, None] * target + sqrt_1mac[t, None, None, None] * noise

            noise_pred = dit(noisy, t, cond=cond)
            loss = F.mse_loss(noise_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(dit.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            batches += 1

        scheduler.step()
        avg_loss = total_loss / batches
        if avg_loss < best_loss:
            best_loss = avg_loss

        if (epoch + 1) % 50 == 0 or epoch == 0 or epoch < 10:
            elapsed = time.time() - t0
            print(f"  Epoch {epoch+1}/{epochs}: loss={avg_loss:.6f}, best={best_loss:.6f}, "
                  f"lr={scheduler.get_last_lr()[0]:.2e} [{elapsed:.1f}s]")

    elapsed = time.time() - t0
    print(f"Diffusion done in {elapsed:.1f}s ({epochs/elapsed:.1f} epochs/s)")
    return dit


# ─── Inference ────────────────────────────────────────────────────────

@torch.no_grad()
def generate_direct(predictor, vae, initial_latent, num_frames=30):
    """Generate frames with direct predictor."""
    predictor.eval()
    latents = [initial_latent]
    for _ in range(num_frames):
        latents.append(predictor(latents[-1]))
    # Decode all
    all_lat = torch.cat(latents, dim=0)
    frames = vae.decode(all_lat)
    return frames  # (N+1, 3, H, W)


@torch.no_grad()
def generate_diffusion(dit, vae, initial_latent, num_frames=30, steps=50, num_timesteps=1000):
    """Generate frames with diffusion model."""
    dit.eval()
    device = initial_latent.device

    betas = torch.linspace(0.0001, 0.02, num_timesteps, device=device)
    alphas = 1.0 - betas
    ac = torch.cumprod(alphas, dim=0)

    step_size = num_timesteps // steps
    timesteps = list(range(num_timesteps - 1, -1, -step_size))

    latents = [initial_latent]
    current = initial_latent

    for _ in range(num_frames):
        x = torch.randn_like(current)
        for i, t in enumerate(timesteps):
            t_t = torch.tensor([t], device=device)
            noise_pred = dit(x, t_t, cond=current)
            a = ac[t]
            a_prev = ac[timesteps[i+1]] if i+1 < len(timesteps) else torch.tensor(1.0, device=device)
            x0 = (x - (1-a).sqrt() * noise_pred) / a.sqrt()
            x0 = x0.clamp(-3, 3)
            if i+1 < len(timesteps):
                x = a_prev.sqrt() * x0 + (1-a_prev).sqrt() * noise_pred
            else:
                x = x0
        latents.append(x)
        current = x

    all_lat = torch.cat(latents, dim=0)
    frames = vae.decode(all_lat)
    return frames


def save_frames(frames, output_dir, prefix="frame"):
    """Save frames to disk."""
    import numpy as np
    from PIL import Image

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    frames_np = frames.float().cpu().numpy()

    for i, f in enumerate(frames_np):
        img = (f.transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(img).save(Path(output_dir) / f"{prefix}_{i:03d}.png")

    # Save GIF
    from PIL import Image as PILImage
    pil_frames = []
    for f in frames_np:
        img = (f.transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
        pil_frames.append(PILImage.fromarray(img))
    pil_frames[0].save(
        Path(output_dir) / "animation.gif",
        save_all=True, append_images=pil_frames[1:], duration=100, loop=0,
    )
    print(f"Saved {len(frames_np)} frames + GIF to {output_dir}")


# ─── Main ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_videos", type=int, default=100)
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--vae_epochs", type=int, default=50)
    parser.add_argument("--direct_epochs", type=int, default=1000)
    parser.add_argument("--diffusion_epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--generate_frames", type=int, default=30)
    parser.add_argument("--skip_diffusion", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        print(f"GPU: {free/1e9:.1f}GB free / {total/1e9:.1f}GB total")

    t_total = time.time()

    # ─── Step 1: VAE ─────────────────────────────────
    vae = SimpleVAE().to(device)
    vae_path = Path("checkpoints/world_model/vae.pt")

    if vae_path.exists():
        print(f"Loading pretrained VAE from {vae_path}")
        vae.load_state_dict(torch.load(vae_path, map_location=device))
    else:
        all_frames = load_all_video_frames(args.num_videos, args.num_frames, args.resolution, device)
        vae = train_vae_fast(vae, all_frames, epochs=args.vae_epochs, batch_size=args.batch_size)
        vae_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(vae.state_dict(), vae_path)
        del all_frames
        torch.cuda.empty_cache()

    vae.eval()

    # ─── Step 2: Precompute latent pairs ─────────────
    pairs = load_all_video_latents(
        vae, args.num_videos, args.num_frames, args.resolution, device
    )

    # ─── Step 3: Train direct predictor ──────────────
    predictor = DirectPredictor(channels=4, dim=128).to(device)
    n_params = sum(p.numel() for p in predictor.parameters())
    print(f"\nDirect predictor: {n_params:,} params")

    predictor = train_direct_fast(
        predictor, pairs,
        epochs=args.direct_epochs, batch_size=args.batch_size, lr=3e-4,
    )

    # Save
    ckpt_dir = Path("checkpoints/fast")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(predictor.state_dict(), ckpt_dir / "direct_predictor.pt")

    # ─── Step 4: Train diffusion ─────────────────────
    if not args.skip_diffusion:
        dit = ConditionalDiT(channels=4, dim=512, heads=8, layers=6, patch=4).to(device)
        n_params = sum(p.numel() for p in dit.parameters())
        print(f"\nConditional DiT: {n_params:,} params")

        dit = train_diffusion_fast(
            dit, pairs,
            epochs=args.diffusion_epochs, batch_size=args.batch_size, lr=1e-4,
        )
        torch.save(dit.state_dict(), ckpt_dir / "conditional_dit.pt")

    # ─── Step 5: Generate demo frames ────────────────
    print("\n=== Generating Demo Frames ===")
    initial_latent = pairs[0, 0].unsqueeze(0)  # First frame of first video

    # Direct predictor output
    frames_direct = generate_direct(predictor, vae, initial_latent, args.generate_frames)
    save_frames(frames_direct, "inference_output/fast_direct")

    if not args.skip_diffusion:
        frames_diff = generate_diffusion(dit, vae, initial_latent, args.generate_frames, steps=50)
        save_frames(frames_diff, "inference_output/fast_diffusion")

    # Save ground truth (first video)
    gt_latents = pairs[:args.num_frames-1, 0]  # first video's frames
    gt_frames = vae.decode(gt_latents[:args.generate_frames+1])
    save_frames(gt_frames, "inference_output/fast_ground_truth")

    elapsed = time.time() - t_total
    print(f"\n{'='*60}")
    print(f"ALL DONE in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
