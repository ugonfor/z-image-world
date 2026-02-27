#!/usr/bin/env python3
"""
Inference Script for World Model

Generate frames autoregressively from a trained world model.

Usage:
    # Generate from a saved image
    uv run python scripts/inference_world_model.py \
        --checkpoint checkpoints/world_model/world_model.pt \
        --input input.png \
        --num_frames 30

    # Generate from a video frame
    uv run python scripts/inference_world_model.py \
        --checkpoint checkpoints/world_model/world_model.pt \
        --video data/videos/synthetic_video_000.mp4 \
        --num_frames 30
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class WorldModelInference:
    """Inference class for the trained World Model."""

    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Load checkpoint
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        config = checkpoint.get('config', {})
        self.resolution = config.get('resolution', 256)

        # Initialize models
        from scripts.train_world_model import SimpleVAE, SimpleCausalDiT

        self.vae = SimpleVAE(
            latent_channels=config.get('latent_channels', 4),
            base_channels=64
        ).to(self.device)

        self.dit = SimpleCausalDiT(
            in_channels=config.get('latent_channels', 4),
            hidden_dim=config.get('hidden_dim', 512),
            num_heads=8,
            num_layers=config.get('num_layers', 6),
            patch_size=4,
        ).to(self.device)

        # Load weights
        self.vae.load_state_dict(checkpoint['vae_state_dict'])
        self.dit.load_state_dict(checkpoint['dit_state_dict'])

        self.vae.eval()
        self.dit.eval()

        # Setup noise schedule
        self.num_timesteps = 1000
        betas = torch.linspace(0.0001, 0.02, self.num_timesteps, device=self.device)
        alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)

        print("Model loaded successfully!")

    def load_image(self, image_path: str) -> torch.Tensor:
        """Load an image file."""
        import imageio
        import cv2

        img = imageio.imread(image_path)
        if img.shape[2] == 4:  # RGBA
            img = img[:, :, :3]

        # Resize
        img = cv2.resize(img, (self.resolution, self.resolution))

        # Convert to tensor
        img = torch.from_numpy(img).float() / 255.0
        img = img.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)

        return img.to(self.device)

    def load_video_frame(self, video_path: str, frame_idx: int = 0) -> torch.Tensor:
        """Load a frame from a video file."""
        import imageio
        import cv2

        reader = imageio.get_reader(video_path)
        frame = reader.get_data(frame_idx)
        reader.close()

        # Resize
        frame = cv2.resize(frame, (self.resolution, self.resolution))

        # Convert to tensor
        frame = torch.from_numpy(frame).float() / 255.0
        frame = frame.permute(2, 0, 1).unsqueeze(0)

        return frame.to(self.device)

    @torch.no_grad()
    def generate(
        self,
        initial_frame: torch.Tensor,
        num_frames: int = 30,
        num_inference_steps: int = 20,
        verbose: bool = True,
    ) -> list[torch.Tensor]:
        """Generate frames autoregressively.

        Args:
            initial_frame: Starting frame (1, 3, H, W) in [0, 1]
            num_frames: Number of frames to generate
            num_inference_steps: Denoising steps per frame
            verbose: Print progress

        Returns:
            List of generated frames
        """
        # Timesteps for inference
        step_size = self.num_timesteps // num_inference_steps
        timesteps = list(range(self.num_timesteps - 1, -1, -step_size))

        # Encode initial frame
        current_latent, _, _ = self.vae.encode(initial_frame)
        generated_frames = [initial_frame.cpu()]

        start_time = time.perf_counter()

        for frame_idx in range(num_frames):
            if verbose:
                print(f"  Generating frame {frame_idx + 1}/{num_frames}...", end='\r')

            # Start from noise
            latent = torch.randn_like(current_latent)

            # Denoise, conditioned on current frame
            for i, t in enumerate(timesteps):
                t_tensor = torch.tensor([t], device=self.device)

                # Predict noise conditioned on current frame
                noise_pred = self.dit(latent, t_tensor, cond=current_latent)

                # DDPM step
                alpha = self.alphas_cumprod[t]
                alpha_prev = self.alphas_cumprod[timesteps[i + 1]] if i + 1 < len(timesteps) else torch.tensor(1.0)

                # Predict x0
                sqrt_alpha = torch.sqrt(alpha)
                sqrt_one_minus_alpha = torch.sqrt(1 - alpha)
                x0_pred = (latent - sqrt_one_minus_alpha * noise_pred) / sqrt_alpha
                x0_pred = x0_pred.clamp(-3, 3)

                # Compute x_{t-1}
                if i + 1 < len(timesteps):
                    sqrt_alpha_prev = torch.sqrt(alpha_prev)
                    sqrt_one_minus_alpha_prev = torch.sqrt(1 - alpha_prev)
                    latent = sqrt_alpha_prev * x0_pred + sqrt_one_minus_alpha_prev * noise_pred
                else:
                    latent = x0_pred

            # Decode for visualization
            frame = self.vae.decode(latent)
            generated_frames.append(frame.cpu())

            # Stay in latent space for next frame's condition (avoid VAE roundtrip error)
            current_latent = latent.detach()

        elapsed = time.perf_counter() - start_time
        fps = num_frames / elapsed

        if verbose:
            print(f"\nGenerated {num_frames} frames in {elapsed:.2f}s ({fps:.2f} FPS)")

        return generated_frames

    def save_frames(self, frames: list[torch.Tensor], output_dir: Path):
        """Save frames as images."""
        import imageio

        output_dir.mkdir(parents=True, exist_ok=True)

        for i, frame in enumerate(frames):
            if frame.dim() == 4:
                frame = frame[0]
            frame = frame.permute(1, 2, 0).numpy()
            frame = (frame * 255).clip(0, 255).astype(np.uint8)

            output_path = output_dir / f"frame_{i:03d}.png"
            imageio.imwrite(str(output_path), frame)

        print(f"Saved {len(frames)} frames to {output_dir}")

    def save_video(self, frames: list[torch.Tensor], output_path: Path, fps: int = 10):
        """Save frames as video."""
        import imageio

        video_frames = []
        for frame in frames:
            if frame.dim() == 4:
                frame = frame[0]
            frame = frame.permute(1, 2, 0).numpy()
            frame = (frame * 255).clip(0, 255).astype(np.uint8)
            video_frames.append(frame)

        imageio.mimwrite(str(output_path), video_frames, fps=fps)
        print(f"Saved video to {output_path}")

    def save_gif(self, frames: list[torch.Tensor], output_path: Path, fps: int = 10):
        """Save frames as GIF."""
        import imageio

        gif_frames = []
        for frame in frames:
            if frame.dim() == 4:
                frame = frame[0]
            frame = frame.permute(1, 2, 0).numpy()
            frame = (frame * 255).clip(0, 255).astype(np.uint8)
            gif_frames.append(frame)

        duration = 1000 / fps  # milliseconds per frame
        imageio.mimwrite(str(output_path), gif_frames, duration=duration, loop=0)
        print(f"Saved GIF to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="World Model Inference")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/world_model/world_model.pt",
                       help="Path to model checkpoint")
    parser.add_argument("--input", type=str, default=None, help="Input image path")
    parser.add_argument("--video", type=str, default=None, help="Input video path (uses first frame)")
    parser.add_argument("--frame_idx", type=int, default=0, help="Frame index from video")
    parser.add_argument("--num_frames", type=int, default=30, help="Number of frames to generate")
    parser.add_argument("--inference_steps", type=int, default=20, help="Denoising steps per frame")
    parser.add_argument("--output_dir", type=str, default="inference_output", help="Output directory")
    parser.add_argument("--save_video", action="store_true", help="Save as video")
    parser.add_argument("--save_gif", action="store_true", help="Save as GIF")
    parser.add_argument("--fps", type=int, default=10, help="Output video/GIF FPS")
    args = parser.parse_args()

    # Initialize model
    model = WorldModelInference(args.checkpoint)

    # Load input
    if args.input:
        initial_frame = model.load_image(args.input)
        print(f"Loaded image: {args.input}")
    elif args.video:
        initial_frame = model.load_video_frame(args.video, args.frame_idx)
        print(f"Loaded frame {args.frame_idx} from video: {args.video}")
    else:
        # Default: load from synthetic video
        default_video = Path("data/videos/synthetic_video_000.mp4")
        if default_video.exists():
            initial_frame = model.load_video_frame(str(default_video), 0)
            print(f"Loaded default video frame: {default_video}")
        else:
            # Generate random initial frame
            print("No input specified, using random initial frame")
            initial_frame = torch.rand(1, 3, model.resolution, model.resolution, device=model.device)

    # Generate frames
    print(f"\nGenerating {args.num_frames} frames...")
    frames = model.generate(
        initial_frame,
        num_frames=args.num_frames,
        num_inference_steps=args.inference_steps,
    )

    # Save outputs
    output_dir = Path(args.output_dir)

    # Save individual frames
    model.save_frames(frames, output_dir / "frames")

    # Save video
    if args.save_video:
        model.save_video(frames, output_dir / "generated.mp4", fps=args.fps)

    # Save GIF
    if args.save_gif:
        model.save_gif(frames, output_dir / "generated.gif", fps=args.fps)

    # Always save a small preview GIF
    model.save_gif(frames, output_dir / "preview.gif", fps=args.fps)

    print("\n" + "="*60)
    print("INFERENCE COMPLETE")
    print("="*60)
    print(f"Output directory: {output_dir}")
    print(f"Generated frames: {len(frames)}")
    print("="*60)


if __name__ == "__main__":
    main()
