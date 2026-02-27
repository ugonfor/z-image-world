#!/usr/bin/env python3
"""Quick evaluation of trained models from train_fast.py."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import numpy as np
from PIL import Image


# Copy model definitions from train_fast.py (needed for loading)
class DirectPredictor(nn.Module):
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
        return mu, mu, logvar  # Use mu directly (no sampling for eval)

    def decode(self, z):
        return self.decoder(z)


def generate_training_video(vid_idx=0, num_frames=16, resolution=256):
    """Generate a synthetic video using the SAME logic as training."""
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

        frame = np.stack([pattern * c for c in color], axis=2)  # (H, W, 3)
        noise = np.random.rand(h, w, 3).astype(np.float32) * 0.08 - 0.04
        frame = np.clip(frame + noise, 0, 1).astype(np.float32)
        frames.append(frame)

    print(f"  Video {vid_idx}: motion={motion}, color=[{color[0]:.2f},{color[1]:.2f},{color[2]:.2f}]")
    return frames


@torch.no_grad()
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load VAE
    print("Loading VAE...")
    vae = SimpleVAE().to(device)
    vae_ckpt = torch.load("checkpoints/world_model/vae.pt", map_location=device)
    if "model_state_dict" in vae_ckpt:
        vae.load_state_dict(vae_ckpt["model_state_dict"])
    else:
        vae.load_state_dict(vae_ckpt)
    vae.eval()

    # Load predictor
    print("Loading DirectPredictor...")
    predictor = DirectPredictor(channels=4, dim=128).to(device)
    predictor.load_state_dict(torch.load("checkpoints/fast/direct_predictor.pt", map_location=device))
    predictor.eval()

    # Test on multiple training videos
    for vid_idx in [0, 5, 10, 42]:
        print(f"\n=== Evaluating video {vid_idx} ===")
        frames = generate_training_video(vid_idx=vid_idx, num_frames=16, resolution=256)

        # Encode first frame
        first_frame = torch.from_numpy(frames[0]).permute(2, 0, 1).unsqueeze(0).to(device)
        latent, _, _ = vae.encode(first_frame)

        # Generate 15 frames autoregressively (matching training video length)
        out_dir = Path(f"inference_output/eval_v{vid_idx}")
        out_dir.mkdir(parents=True, exist_ok=True)

        generated = [first_frame.cpu()]
        current = latent

        for i in range(15):
            current = predictor(current)
            decoded = vae.decode(current)
            generated.append(decoded.cpu())

        # Save frames and GIF
        pil_frames = []
        for i, frame in enumerate(generated):
            img = (frame[0].permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
            Image.fromarray(img).save(out_dir / f"frame_{i:03d}.png")
            pil_frames.append(Image.fromarray(img))
        pil_frames[0].save(out_dir / "animation.gif", save_all=True, append_images=pil_frames[1:],
                           duration=100, loop=0)

        # Ground truth
        gt_dir = Path(f"inference_output/eval_gt_v{vid_idx}")
        gt_dir.mkdir(parents=True, exist_ok=True)
        gt_pil = []
        for i, f in enumerate(frames):
            img = (f * 255).clip(0, 255).astype(np.uint8)
            Image.fromarray(img).save(gt_dir / f"frame_{i:03d}.png")
            gt_pil.append(Image.fromarray(img))
        gt_pil[0].save(gt_dir / "animation.gif", save_all=True, append_images=gt_pil[1:],
                       duration=100, loop=0)

        # PSNR
        print("  Frame PSNR vs ground truth:")
        for i in range(min(6, len(frames))):
            gt = torch.from_numpy(frames[i]).permute(2, 0, 1).unsqueeze(0)
            pred = generated[i]
            mse = F.mse_loss(pred, gt).item()
            psnr = -10 * np.log10(mse + 1e-8)
            print(f"    Frame {i}: PSNR={psnr:.1f} dB")

        # Also test: VAE roundtrip quality
        if vid_idx == 0:
            print("  VAE roundtrip quality:")
            for i in range(min(3, len(frames))):
                f_tensor = torch.from_numpy(frames[i]).permute(2, 0, 1).unsqueeze(0).to(device)
                z, _, _ = vae.encode(f_tensor)
                recon = vae.decode(z).cpu()
                gt = f_tensor.cpu()
                mse = F.mse_loss(recon, gt).item()
                psnr = -10 * np.log10(mse + 1e-8)
                print(f"    Frame {i}: PSNR={psnr:.1f} dB")


if __name__ == "__main__":
    import torch.nn.functional as F
    main()
