#!/usr/bin/env python3
"""
Z-Image World Model Inference

Generate frames using the pretrained Z-Image world model.
Even without training, the model can generate images (as pure Z-Image).
After training, it generates temporally coherent frame sequences.

Usage:
    # Generate frames from noise (no training needed)
    uv run python scripts/inference_zimage_world.py --num_frames 8

    # Generate from a starting image
    uv run python scripts/inference_zimage_world.py --input image.png --num_frames 16

    # Load trained checkpoint
    uv run python scripts/inference_zimage_world.py --checkpoint checkpoints/zimage_world/world_model_final.pt

    # With action sequence
    uv run python scripts/inference_zimage_world.py --actions forward,forward,left,forward
"""

import argparse
import os
import sys
import time
from pathlib import Path

sys.stdout = os.fdopen(sys.stdout.fileno(), "w", buffering=1)
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from PIL import Image
from einops import rearrange


ACTION_MAP = {
    "forward": 0, "backward": 1, "left": 2, "right": 3,
    "forward_left": 4, "forward_right": 5, "backward_left": 6, "backward_right": 7,
    "idle": 8, "look_up": 9, "look_down": 10, "look_left": 11, "look_right": 12,
    "jump": 13, "crouch": 14, "interact": 15, "attack": 16,
}


def ddpm_denoise(model, noisy_latent, num_steps=4, device="cuda"):
    """Simple DDPM denoising loop."""
    # Linearly spaced timesteps from 999 to 0
    timesteps = torch.linspace(999, 0, num_steps + 1, device=device).long()

    x = noisy_latent
    for i in range(num_steps):
        t = timesteps[i].float().unsqueeze(0)
        t_next = timesteps[i + 1].float().unsqueeze(0)

        with torch.no_grad():
            # Model predicts v (velocity)
            v_pred = model(x, t)

        # Simple step: move towards prediction
        alpha = 1.0 - t / 1000.0
        alpha_next = 1.0 - t_next / 1000.0
        # Simplified DDPM step
        x = x * (alpha_next / alpha).sqrt().view(-1, 1, 1, 1) + v_pred * (
            (1 - alpha_next).sqrt() - (1 - alpha).sqrt() * (alpha_next / alpha).sqrt()
        ).view(-1, 1, 1, 1)

    return x


def generate_frames(model, args, device):
    """Generate a sequence of frames."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    resolution = args.resolution
    latent_h, latent_w = resolution // 8, resolution // 8
    num_frames = args.num_frames

    # Parse actions
    actions = None
    if args.actions:
        action_names = args.actions.split(",")
        action_ids = [ACTION_MAP.get(a.strip(), 8) for a in action_names]
        # Pad or truncate to num_frames
        while len(action_ids) < num_frames:
            action_ids.append(action_ids[-1] if action_ids else 8)
        action_ids = action_ids[:num_frames]
        actions = torch.tensor([action_ids], device=device)
        print(f"Actions: {action_names[:num_frames]}")

    # Load starting image if provided
    start_latent = None
    if args.input:
        print(f"Loading starting image: {args.input}")
        img = Image.open(args.input).convert("RGB").resize((resolution, resolution))
        img_tensor = torch.from_numpy(np.array(img)).float() / 255.0
        img_tensor = rearrange(img_tensor, "h w c -> 1 1 c h w").to(device, dtype=torch.bfloat16)
        with torch.no_grad():
            start_latent = model.encode_frames(img_tensor)  # (1, 1, 16, H//8, W//8)

    # Setup Flow Matching Euler schedule (matching Z-Image's scheduler)
    # Z-Image uses FlowMatchEulerDiscreteScheduler with shift=3.0
    num_denoise_steps = args.denoise_steps

    # Compute sigmas with shift (matching Z-Image)
    shift = 3.0
    base_sigmas = torch.linspace(1.0, 0.0, num_denoise_steps + 1, device=device)
    sigmas = shift * base_sigmas / (1 + (shift - 1) * base_sigmas)
    # Corresponding DDPM-style timesteps (sigmas * 1000)
    denoise_timesteps = (sigmas[:-1] * 1000).long()

    def flow_match_denoise_frame(model, context_latents, noise, action_seq):
        """Flow Matching Euler denoising (matching Z-Image)."""
        x_t = noise  # (1, 16, H, W)
        ctx_len = context_latents.shape[1] if context_latents is not None else 0

        for step_idx in range(num_denoise_steps):
            sigma = sigmas[step_idx]
            sigma_next = sigmas[step_idx + 1]
            dt = sigma_next - sigma  # Negative (sigma decreasing)

            # Z-Image timestep format: DDPM-style, model handles normalization
            t_val = denoise_timesteps[step_idx].float()

            # Build sequence: [context (clean, t=0) | current (noisy)]
            if context_latents is not None:
                full_seq = torch.cat([context_latents, x_t.unsqueeze(1)], dim=1)
                t_ctx = torch.zeros(1, ctx_len, device=device)
                t_cur = t_val.unsqueeze(0).unsqueeze(0)
                timesteps = torch.cat([t_ctx, t_cur], dim=1)
            else:
                full_seq = x_t.unsqueeze(1)
                timesteps = t_val.unsqueeze(0).unsqueeze(0)

            # Forward pass: model predicts flow matching velocity
            velocity = model(full_seq, timesteps, actions=action_seq)
            if velocity.dim() == 5:
                velocity = velocity[:, -1]

            # Euler step: x_{t+dt} = x_t + dt * velocity
            x_t = x_t + dt * velocity

        return x_t

    # Generate frames autoregressively
    print(f"\nGenerating {num_frames} frames at {resolution}x{resolution} ({num_denoise_steps}-step Flow Match Euler)...")
    generated_latents = []
    t_start = time.time()

    for f in range(num_frames):
        # Start from previous latent or noise
        if f == 0 and start_latent is not None:
            generated_latents.append(start_latent[:, 0])  # (1, 16, H, W)
            continue

        # Generate noise for new frame
        noise = torch.randn(1, 16, latent_h, latent_w, device=device, dtype=torch.bfloat16)

        # Context: use previous frames
        context_latents = None
        if generated_latents:
            context = generated_latents[-args.context_frames:]
            context_latents = torch.stack(context, dim=1)  # (1, C, 16, H, W)

        # Actions for context window
        frame_actions = None
        if actions is not None and context_latents is not None:
            ctx_len = context_latents.shape[1]
            start_idx = max(0, f - ctx_len)
            frame_actions = actions[:, start_idx : f + 1]
            if frame_actions.shape[1] < ctx_len + 1:
                pad = torch.full((1, ctx_len + 1 - frame_actions.shape[1]), 8, device=device)
                frame_actions = torch.cat([pad, frame_actions], dim=1)

        # Denoise using Flow Matching Euler steps
        with torch.no_grad():
            clean_latent = flow_match_denoise_frame(model, context_latents, noise, frame_actions)

        generated_latents.append(clean_latent)

        if (f + 1) % 5 == 0 or f == num_frames - 1:
            elapsed = time.time() - t_start
            print(f"  Frame {f + 1}/{num_frames} ({elapsed:.1f}s)")

    elapsed = time.time() - t_start
    print(f"Generated {num_frames} frames in {elapsed:.1f}s ({num_frames / elapsed:.1f} FPS)")

    # Decode all latents to images
    print("\nDecoding latents to images...")
    all_latents = torch.stack(generated_latents, dim=1)  # (1, F, 16, H, W)
    with torch.no_grad():
        images = model.decode_latents(all_latents)  # (1, F, 3, H, W)

    # Save individual frames
    images_np = images[0].float().cpu().numpy()  # (F, 3, H, W)
    for f in range(num_frames):
        img = (images_np[f].transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(img).save(output_dir / f"frame_{f:04d}.png")

    print(f"Saved {num_frames} frames to {output_dir}/")

    # Save as GIF if requested
    if args.save_gif:
        pil_frames = []
        for f in range(num_frames):
            img = (images_np[f].transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
            pil_frames.append(Image.fromarray(img))

        gif_path = output_dir / "generation.gif"
        pil_frames[0].save(
            gif_path, save_all=True, append_images=pil_frames[1:],
            duration=100, loop=0,
        )
        print(f"Saved GIF: {gif_path}")

    # Save as video if requested
    if args.save_video:
        try:
            import imageio
            video_path = output_dir / "generation.mp4"
            writer = imageio.get_writer(str(video_path), fps=10)
            for f in range(num_frames):
                img = (images_np[f].transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
                writer.append_data(img)
            writer.close()
            print(f"Saved video: {video_path}")
        except ImportError:
            print("imageio not installed, skipping video output")


def main():
    parser = argparse.ArgumentParser(description="Z-Image World Model Inference")
    parser.add_argument("--num_frames", type=int, default=8, help="Number of frames to generate")
    parser.add_argument("--resolution", type=int, default=512, help="Image resolution")
    parser.add_argument("--input", type=str, default=None, help="Starting image path")
    parser.add_argument("--actions", type=str, default=None, help="Comma-separated action sequence")
    parser.add_argument("--checkpoint", type=str, default=None, help="Trained checkpoint path")
    parser.add_argument("--temporal_every_n", type=int, default=1, help="Temporal attention frequency")
    parser.add_argument("--denoise_steps", type=int, default=4, help="Number of DDIM denoising steps")
    parser.add_argument("--context_frames", type=int, default=3, help="Number of context frames")
    parser.add_argument("--output_dir", type=str, default="inference_output/zimage_world", help="Output directory")
    parser.add_argument("--save_gif", action="store_true", help="Save as GIF")
    parser.add_argument("--save_video", action="store_true", help="Save as MP4")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load model
    print("\n=== Loading Z-Image World Model ===")
    from models.zimage_world_model import ZImageWorldModel

    model = ZImageWorldModel.from_pretrained(
        "Tongyi-MAI/Z-Image-Turbo",
        torch_dtype=torch.bfloat16,
        temporal_every_n=args.temporal_every_n,
        freeze_spatial=True,
        device=device,
    )

    # Load trained weights if provided
    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.temporal_layers.load_state_dict(ckpt["temporal_state_dict"])
        model.action_injections.load_state_dict(ckpt["action_injections_state_dict"])
        model.action_encoder.load_state_dict(ckpt["action_encoder_state_dict"])
        print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")

    model.eval()

    generate_frames(model, args, device)

    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info(0)
        print(f"\nGPU Memory: {free / 1e9:.1f}GB free / {total / 1e9:.1f}GB total")


if __name__ == "__main__":
    main()
