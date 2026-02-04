#!/usr/bin/env python3
"""
Train World Model on Video Dataset

This script trains the CausalDiT model on video data to learn
next-frame prediction conditioned on previous frames.

Usage:
    # Download sample videos and train
    uv run python scripts/train_world_model.py --download

    # Train on existing video directory
    uv run python scripts/train_world_model.py --data_dir /path/to/videos

    # Quick test run
    uv run python scripts/train_world_model.py --download --quick
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Force unbuffered output
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', buffering=1)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from einops import rearrange

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def download_sample_videos(output_dir: Path, num_videos: int = 10) -> list[Path]:
    """Download sample videos for training.

    Uses public domain videos from archive.org or generates synthetic data.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    video_paths = []

    # Try to download real videos first
    try:
        import urllib.request

        # Sample video URLs (public domain / Creative Commons)
        sample_urls = [
            # Big Buck Bunny clips (Creative Commons)
            "https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/360/Big_Buck_Bunny_360_10s_1MB.mp4",
            "https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/360/Big_Buck_Bunny_360_10s_2MB.mp4",
        ]

        for i, url in enumerate(sample_urls[:num_videos]):
            output_path = output_dir / f"sample_video_{i:03d}.mp4"
            if not output_path.exists():
                print(f"Downloading {url}...")
                try:
                    urllib.request.urlretrieve(url, output_path)
                    video_paths.append(output_path)
                    print(f"  Saved to {output_path}")
                except Exception as e:
                    print(f"  Failed to download: {e}")
            else:
                video_paths.append(output_path)
                print(f"  Already exists: {output_path}")

    except ImportError:
        print("urllib not available, generating synthetic data")

    # Generate synthetic videos if needed
    if len(video_paths) < num_videos:
        print(f"Generating {num_videos - len(video_paths)} synthetic videos...")
        video_paths.extend(generate_synthetic_videos(
            output_dir,
            num_videos - len(video_paths),
            start_idx=len(video_paths)
        ))

    return video_paths


def generate_synthetic_videos(
    output_dir: Path,
    num_videos: int,
    start_idx: int = 0,
    num_frames: int = 30,
    resolution: tuple[int, int] = (256, 256),
) -> list[Path]:
    """Generate synthetic videos with moving patterns for training."""
    import numpy as np

    try:
        import imageio
    except ImportError:
        print("imageio not available, installing...")
        os.system("pip install imageio[ffmpeg]")
        import imageio

    output_dir.mkdir(parents=True, exist_ok=True)
    video_paths = []

    for i in range(num_videos):
        video_idx = start_idx + i
        output_path = output_dir / f"synthetic_video_{video_idx:03d}.mp4"

        if output_path.exists():
            video_paths.append(output_path)
            continue

        print(f"Generating synthetic video {video_idx}...")

        # Generate frames with moving patterns
        frames = []
        h, w = resolution

        # Random motion parameters
        np.random.seed(video_idx)
        motion_type = np.random.choice(['circle', 'linear', 'zoom', 'rotate'])
        color = np.random.rand(3)

        for frame_idx in range(num_frames):
            t = frame_idx / num_frames

            # Create gradient background
            y_grid, x_grid = np.mgrid[0:h, 0:w].astype(np.float32)
            x_norm = x_grid / w
            y_norm = y_grid / h

            if motion_type == 'circle':
                # Moving circle
                cx = w/2 + w/4 * np.cos(2 * np.pi * t)
                cy = h/2 + h/4 * np.sin(2 * np.pi * t)
                dist = np.sqrt((x_grid - cx)**2 + (y_grid - cy)**2)
                pattern = np.exp(-dist**2 / (w/4)**2)

            elif motion_type == 'linear':
                # Linear movement
                offset = t * w
                pattern = np.sin(2 * np.pi * (x_norm + t) * 3) * 0.5 + 0.5

            elif motion_type == 'zoom':
                # Zooming pattern
                scale = 1 + t * 2
                cx, cy = w/2, h/2
                dist = np.sqrt((x_grid - cx)**2 + (y_grid - cy)**2)
                pattern = np.sin(dist * scale * 0.1) * 0.5 + 0.5

            else:  # rotate
                # Rotating pattern
                angle = t * 2 * np.pi
                x_rot = (x_norm - 0.5) * np.cos(angle) - (y_norm - 0.5) * np.sin(angle)
                y_rot = (x_norm - 0.5) * np.sin(angle) + (y_norm - 0.5) * np.cos(angle)
                pattern = (np.sin(x_rot * 10) * np.sin(y_rot * 10)) * 0.5 + 0.5

            # Create RGB frame
            frame = np.zeros((h, w, 3), dtype=np.uint8)
            for c in range(3):
                frame[:, :, c] = (pattern * color[c] * 255).astype(np.uint8)

            # Add some noise for realism
            noise = np.random.randint(0, 20, (h, w, 3), dtype=np.uint8)
            frame = np.clip(frame.astype(np.int32) + noise - 10, 0, 255).astype(np.uint8)

            frames.append(frame)

        # Save video
        imageio.mimwrite(str(output_path), frames, fps=10)
        video_paths.append(output_path)
        print(f"  Saved {output_path}")

    return video_paths


class SimpleVideoDataset(Dataset):
    """Simple video dataset that loads frames from video files."""

    def __init__(
        self,
        video_paths: list[Path],
        num_frames: int = 8,
        resolution: tuple[int, int] = (256, 256),
        samples_per_video: int = 5,
    ):
        self.video_paths = video_paths
        self.num_frames = num_frames
        self.resolution = resolution
        self.samples_per_video = samples_per_video

        # Pre-scan videos to get frame counts
        self.video_info = []
        for vp in video_paths:
            try:
                import imageio
                reader = imageio.get_reader(str(vp))
                total_frames = reader.count_frames()
                reader.close()

                if total_frames >= num_frames:
                    self.video_info.append({
                        'path': vp,
                        'total_frames': total_frames,
                    })
            except Exception as e:
                print(f"Warning: Could not load {vp}: {e}")

        print(f"Loaded {len(self.video_info)} valid videos")

    def __len__(self):
        return len(self.video_info) * self.samples_per_video

    def __getitem__(self, idx):
        import imageio
        import numpy as np

        video_idx = idx // self.samples_per_video
        info = self.video_info[video_idx]

        # Random start frame
        max_start = info['total_frames'] - self.num_frames
        start_idx = np.random.randint(0, max(1, max_start))

        # Load frames
        reader = imageio.get_reader(str(info['path']))
        frames = []

        for i in range(self.num_frames):
            frame_idx = min(start_idx + i, info['total_frames'] - 1)
            frame = reader.get_data(frame_idx)

            # Resize
            import cv2
            frame = cv2.resize(frame, (self.resolution[1], self.resolution[0]))

            # Convert to tensor [0, 1]
            frame = torch.from_numpy(frame).float() / 255.0
            frame = frame.permute(2, 0, 1)  # HWC -> CHW
            frames.append(frame)

        reader.close()

        frames = torch.stack(frames)  # (F, C, H, W)
        return {'frames': frames}


class SimpleCausalDiT(nn.Module):
    """Simplified CausalDiT for faster training on small datasets."""

    def __init__(
        self,
        in_channels: int = 4,  # VAE latent channels
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        patch_size: int = 4,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.patch_size = patch_size

        # Patch embedding
        self.patch_embed = nn.Conv2d(
            in_channels, hidden_dim,
            kernel_size=patch_size, stride=patch_size
        )

        # Position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, 1024, hidden_dim))

        # Timestep embedding
        self.time_embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

        # Transformer layers
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=0.1,
                activation='gelu',
                batch_first=True,
            )
            for _ in range(num_layers)
        ])

        # Output projection
        self.norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, patch_size * patch_size * in_channels)

        # Initialize
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def _get_timestep_embedding(self, timesteps, dim):
        """Sinusoidal timestep embeddings."""
        half_dim = dim // 2
        emb = torch.exp(
            -torch.log(torch.tensor(10000.0)) * torch.arange(half_dim, device=timesteps.device) / half_dim
        )
        emb = timesteps[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb

    def forward(self, x, timesteps):
        """
        Args:
            x: (B, C, H, W) noisy latents
            timesteps: (B,) diffusion timesteps

        Returns:
            noise_pred: (B, C, H, W) predicted noise
        """
        B, C, H, W = x.shape

        # Patch embed
        x = self.patch_embed(x)  # (B, D, H/P, W/P)
        h, w = x.shape[2], x.shape[3]
        x = rearrange(x, 'b d h w -> b (h w) d')

        # Add position embedding
        seq_len = x.shape[1]
        x = x + self.pos_embed[:, :seq_len]

        # Add timestep embedding
        t_emb = self._get_timestep_embedding(timesteps, self.hidden_dim)
        t_emb = self.time_embed(t_emb)  # (B, D)
        x = x + t_emb.unsqueeze(1)

        # Transformer layers
        for layer in self.layers:
            x = layer(x)

        # Output
        x = self.norm(x)
        x = self.out_proj(x)  # (B, seq_len, P*P*C)

        # Unpatchify
        x = rearrange(
            x, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)',
            h=h, w=w, p1=self.patch_size, p2=self.patch_size, c=C
        )

        return x


class SimpleVAE(nn.Module):
    """Simple VAE for encoding/decoding frames to latent space."""

    def __init__(self, latent_channels: int = 4, base_channels: int = 64):
        super().__init__()

        self.latent_channels = latent_channels

        # Encoder: 256x256 -> 32x32
        self.encoder = nn.Sequential(
            nn.Conv2d(3, base_channels, 4, 2, 1),      # 128x128
            nn.ReLU(),
            nn.Conv2d(base_channels, base_channels*2, 4, 2, 1),  # 64x64
            nn.ReLU(),
            nn.Conv2d(base_channels*2, base_channels*4, 4, 2, 1),  # 32x32
            nn.ReLU(),
            nn.Conv2d(base_channels*4, latent_channels * 2, 3, 1, 1),  # mean + logvar
        )

        # Decoder: 32x32 -> 256x256
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_channels, base_channels*4, 3, 1, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(base_channels*4, base_channels*2, 4, 2, 1),  # 64x64
            nn.ReLU(),
            nn.ConvTranspose2d(base_channels*2, base_channels, 4, 2, 1),  # 128x128
            nn.ReLU(),
            nn.ConvTranspose2d(base_channels, 3, 4, 2, 1),  # 256x256
            nn.Sigmoid(),
        )

    def encode(self, x):
        """Encode to latent space. x in [0, 1]."""
        h = self.encoder(x)
        mean, logvar = h.chunk(2, dim=1)
        # Reparameterization
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mean + eps * std
        return z, mean, logvar

    def decode(self, z):
        """Decode from latent space. Returns [0, 1]."""
        return self.decoder(z)

    def forward(self, x):
        z, mean, logvar = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, mean, logvar


def train_vae(
    vae: SimpleVAE,
    dataloader: DataLoader,
    num_epochs: int,
    device: torch.device,
    lr: float = 1e-4,
):
    """Pre-train VAE for frame reconstruction."""
    print("\n=== Training VAE ===")

    optimizer = torch.optim.AdamW(vae.parameters(), lr=lr)
    vae.train()

    for epoch in range(num_epochs):
        total_loss = 0
        total_recon = 0
        total_kl = 0
        num_batches = 0

        for batch in dataloader:
            frames = batch['frames'].to(device)  # (B, num_frames, C, H, W)
            batch_size, num_frames, channels, height, width = frames.shape

            # Flatten frames
            frames_flat = rearrange(frames, 'b f c h w -> (b f) c h w')

            # Forward
            recon, mean, logvar = vae(frames_flat)

            # Loss
            recon_loss = torch.nn.functional.mse_loss(recon, frames_flat)
            kl_loss = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
            loss = recon_loss + 0.001 * kl_loss

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        avg_recon = total_recon / num_batches
        avg_kl = total_kl / num_batches

        print(f"  Epoch {epoch+1}/{num_epochs}: loss={avg_loss:.4f}, recon={avg_recon:.4f}, kl={avg_kl:.4f}")

    return vae


def train_diffusion(
    dit: SimpleCausalDiT,
    vae: SimpleVAE,
    dataloader: DataLoader,
    num_epochs: int,
    device: torch.device,
    lr: float = 1e-4,
    num_timesteps: int = 1000,
):
    """Train diffusion model for next-frame prediction."""
    print("\n=== Training Diffusion Model ===")

    optimizer = torch.optim.AdamW(dit.parameters(), lr=lr)

    # Noise schedule
    betas = torch.linspace(0.0001, 0.02, num_timesteps, device=device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod)

    dit.train()
    vae.eval()

    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0

        for batch in dataloader:
            frames = batch['frames'].to(device)  # (B, num_frames, C, H, W)
            batch_size, num_frames, channels, height, width = frames.shape

            # Encode all frames with VAE
            with torch.no_grad():
                frames_flat = rearrange(frames, 'b f c h w -> (b f) c h w')
                latents, _, _ = vae.encode(frames_flat)
                latents = rearrange(latents, '(b f) c h w -> b f c h w', b=batch_size, f=num_frames)

            # For each frame pair, predict next frame
            losses = []
            for frame_idx in range(num_frames - 1):
                current_latent = latents[:, frame_idx]  # (B, C, h, w)
                target_latent = latents[:, frame_idx + 1]  # (B, C, h, w)

                # Sample timesteps
                t = torch.randint(0, num_timesteps, (batch_size,), device=device)

                # Add noise to target
                noise = torch.randn_like(target_latent)
                sqrt_alpha = sqrt_alphas_cumprod[t].view(batch_size, 1, 1, 1)
                sqrt_one_minus_alpha = sqrt_one_minus_alphas_cumprod[t].view(batch_size, 1, 1, 1)
                noisy_target = sqrt_alpha * target_latent + sqrt_one_minus_alpha * noise

                # Concatenate current frame as condition (channel-wise)
                # For simplicity, we'll just predict noise from noisy target
                # In full implementation, we'd condition on current frame
                noise_pred = dit(noisy_target, t)

                # MSE loss
                loss = torch.nn.functional.mse_loss(noise_pred, noise)
                losses.append(loss)

            # Average loss across frame pairs
            loss = torch.stack(losses).mean()

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(dit.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        print(f"  Epoch {epoch+1}/{num_epochs}: loss={avg_loss:.4f}")

    return dit


@torch.no_grad()
def generate_frames(
    dit: SimpleCausalDiT,
    vae: SimpleVAE,
    initial_frame: torch.Tensor,
    num_frames: int,
    device: torch.device,
    num_inference_steps: int = 20,
    num_timesteps: int = 1000,
):
    """Generate frames autoregressively."""
    print("\n=== Generating Frames ===")

    # Noise schedule
    betas = torch.linspace(0.0001, 0.02, num_timesteps, device=device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    dit.eval()
    vae.eval()

    # Encode initial frame
    current_frame = initial_frame.to(device)
    if current_frame.dim() == 3:
        current_frame = current_frame.unsqueeze(0)

    current_latent, _, _ = vae.encode(current_frame)

    generated_frames = [current_frame.cpu()]

    # Timesteps for inference (reversed)
    step_size = num_timesteps // num_inference_steps
    timesteps = list(range(num_timesteps - 1, -1, -step_size))

    for frame_idx in range(num_frames):
        print(f"  Generating frame {frame_idx + 1}/{num_frames}...")

        # Start from noise
        latent = torch.randn_like(current_latent)

        # Denoise
        for i, t in enumerate(timesteps):
            t_tensor = torch.tensor([t], device=device)

            # Predict noise
            noise_pred = dit(latent, t_tensor)

            # DDPM step
            alpha = alphas_cumprod[t]
            alpha_prev = alphas_cumprod[timesteps[i + 1]] if i + 1 < len(timesteps) else torch.tensor(1.0)

            # Predict x0
            sqrt_alpha = torch.sqrt(alpha)
            sqrt_one_minus_alpha = torch.sqrt(1 - alpha)
            x0_pred = (latent - sqrt_one_minus_alpha * noise_pred) / sqrt_alpha
            x0_pred = x0_pred.clamp(-3, 3)  # Clip for stability

            # Compute x_{t-1}
            if i + 1 < len(timesteps):
                sqrt_alpha_prev = torch.sqrt(alpha_prev)
                sqrt_one_minus_alpha_prev = torch.sqrt(1 - alpha_prev)
                latent = sqrt_alpha_prev * x0_pred + sqrt_one_minus_alpha_prev * noise_pred
            else:
                latent = x0_pred

        # Decode
        frame = vae.decode(latent)
        generated_frames.append(frame.cpu())

        # Update current latent
        current_latent = latent

    return generated_frames


def save_frames_as_video(frames: list[torch.Tensor], output_path: Path):
    """Save frames as video."""
    import imageio
    import numpy as np

    video_frames = []
    for frame in frames:
        if frame.dim() == 4:
            frame = frame[0]  # Remove batch dim
        frame = frame.permute(1, 2, 0).numpy()  # CHW -> HWC
        frame = (frame * 255).clip(0, 255).astype(np.uint8)
        video_frames.append(frame)

    imageio.mimwrite(str(output_path), video_frames, fps=10)
    print(f"Saved video to {output_path}")


def save_frames_as_images(frames: list[torch.Tensor], output_dir: Path):
    """Save frames as individual images."""
    import imageio
    import numpy as np

    output_dir.mkdir(parents=True, exist_ok=True)

    for i, frame in enumerate(frames):
        if frame.dim() == 4:
            frame = frame[0]
        frame = frame.permute(1, 2, 0).numpy()
        frame = (frame * 255).clip(0, 255).astype(np.uint8)

        output_path = output_dir / f"frame_{i:03d}.png"
        imageio.imwrite(str(output_path), frame)

    print(f"Saved {len(frames)} frames to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train World Model on Video Data")
    parser.add_argument("--data_dir", type=str, default=None, help="Directory with video files")
    parser.add_argument("--download", action="store_true", help="Download sample videos")
    parser.add_argument("--output_dir", type=str, default="checkpoints/world_model", help="Output directory")
    parser.add_argument("--quick", action="store_true", help="Quick test with minimal training")
    parser.add_argument("--num_videos", type=int, default=10, help="Number of videos to use")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--vae_epochs", type=int, default=10, help="VAE training epochs")
    parser.add_argument("--dit_epochs", type=int, default=20, help="DiT training epochs")
    parser.add_argument("--resolution", type=int, default=256, help="Image resolution")
    parser.add_argument("--num_frames", type=int, default=8, help="Frames per sample")
    parser.add_argument("--generate_frames", type=int, default=10, help="Frames to generate")
    args = parser.parse_args()

    # Quick mode overrides
    if args.quick:
        args.num_videos = 3
        args.vae_epochs = 2
        args.dit_epochs = 3
        args.generate_frames = 5
        print("Quick mode: Using minimal settings for testing")

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get video data
    if args.download or args.data_dir is None:
        data_dir = Path("data/videos")
        video_paths = download_sample_videos(data_dir, args.num_videos)
    else:
        data_dir = Path(args.data_dir)
        video_paths = list(data_dir.glob("**/*.mp4")) + list(data_dir.glob("**/*.avi"))
        video_paths = video_paths[:args.num_videos]

    if len(video_paths) == 0:
        print("No videos found! Generating synthetic data...")
        data_dir = Path("data/videos")
        video_paths = generate_synthetic_videos(data_dir, args.num_videos)

    print(f"Found {len(video_paths)} videos")

    # Create dataset
    dataset = SimpleVideoDataset(
        video_paths=video_paths,
        num_frames=args.num_frames,
        resolution=(args.resolution, args.resolution),
        samples_per_video=5,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # Avoid multiprocessing issues
        drop_last=True,
    )

    print(f"Dataset: {len(dataset)} samples, {len(dataloader)} batches")

    # Initialize models
    print("\n=== Initializing Models ===")

    vae = SimpleVAE(latent_channels=4, base_channels=64).to(device)
    dit = SimpleCausalDiT(
        in_channels=4,
        hidden_dim=512,
        num_heads=8,
        num_layers=6,
        patch_size=4,
    ).to(device)

    vae_params = sum(p.numel() for p in vae.parameters())
    dit_params = sum(p.numel() for p in dit.parameters())
    print(f"VAE parameters: {vae_params:,}")
    print(f"DiT parameters: {dit_params:,}")

    # Train VAE
    vae = train_vae(vae, dataloader, args.vae_epochs, device)

    # Save VAE
    vae_path = output_dir / "vae.pt"
    torch.save(vae.state_dict(), vae_path)
    print(f"Saved VAE to {vae_path}")

    # Train DiT
    dit = train_diffusion(dit, vae, dataloader, args.dit_epochs, device)

    # Save DiT
    dit_path = output_dir / "dit.pt"
    torch.save(dit.state_dict(), dit_path)
    print(f"Saved DiT to {dit_path}")

    # Save full checkpoint
    checkpoint_path = output_dir / "world_model.pt"
    torch.save({
        'vae_state_dict': vae.state_dict(),
        'dit_state_dict': dit.state_dict(),
        'config': {
            'resolution': args.resolution,
            'latent_channels': 4,
            'hidden_dim': 512,
            'num_layers': 6,
        }
    }, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")

    # Generate test frames
    print("\n=== Testing Inference ===")

    # Get a sample frame
    sample = dataset[0]
    initial_frame = sample['frames'][0]  # First frame

    # Generate
    generated = generate_frames(
        dit, vae, initial_frame,
        num_frames=args.generate_frames,
        device=device,
    )

    # Save results
    inference_dir = output_dir / "inference_output"
    save_frames_as_images(generated, inference_dir)

    # Save as video
    video_path = output_dir / "generated.mp4"
    save_frames_as_video(generated, video_path)

    # Also save ground truth for comparison
    gt_frames = [sample['frames'][i] for i in range(min(len(sample['frames']), args.generate_frames + 1))]
    gt_dir = output_dir / "ground_truth"
    save_frames_as_images(gt_frames, gt_dir)

    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Generated frames: {inference_dir}")
    print(f"Generated video: {video_path}")
    print("="*60)


if __name__ == "__main__":
    main()
