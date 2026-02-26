#!/usr/bin/env python3
"""
Launch Interactive Z-Image World Model Demo

Loads the Z-Image-Turbo world model and runs the interactive pygame demo
with keyboard-controlled frame generation.

Usage:
    # Run with default settings (balanced quality)
    uv run python scripts/run_interactive_demo.py

    # Fast mode (128x128, 1-step, ~10+ FPS)
    uv run python scripts/run_interactive_demo.py --quality fast

    # High quality (384x384, 4-step, ~2 FPS)
    uv run python scripts/run_interactive_demo.py --quality quality

    # With trained checkpoint
    uv run python scripts/run_interactive_demo.py --checkpoint checkpoints/zimage_world/world_model_final.pt

    # Custom initial scene
    uv run python scripts/run_interactive_demo.py --prompt "a medieval castle courtyard"
"""

import argparse
import sys
import os
from pathlib import Path

sys.stdout = os.fdopen(sys.stdout.fileno(), "w", buffering=1)
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np


def generate_initial_frame(prompt: str, resolution: int, device: str) -> torch.Tensor:
    """Generate an initial frame using Z-Image text-to-image."""
    try:
        from diffusers import ZImagePipeline

        print(f"Generating initial frame: \"{prompt}\"")
        pipe = ZImagePipeline.from_pretrained(
            "Tongyi-MAI/Z-Image-Turbo",
            torch_dtype=torch.bfloat16,
        )
        pipe.to(device)

        with torch.no_grad():
            result = pipe(
                prompt=prompt,
                height=resolution,
                width=resolution,
                num_inference_steps=8,
                guidance_scale=0.0,
            )
            image = result.images[0]

        # Convert PIL to tensor
        image_np = np.array(image).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0)

        # Free the t2i pipeline
        del pipe
        torch.cuda.empty_cache()

        return image_tensor.to(device, dtype=torch.bfloat16)

    except Exception as e:
        print(f"Could not generate initial frame: {e}")
        print("Using random noise as initial frame")
        return torch.rand(1, 3, resolution, resolution, device=device, dtype=torch.bfloat16)


def run_demo(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Check for display
    display_available = os.environ.get("DISPLAY") is not None
    if not display_available:
        print("WARNING: No display detected. Set DISPLAY env var or use X forwarding.")
        print("  export DISPLAY=:0  (for local display)")
        print("  ssh -X user@host   (for X forwarding)")

    from inference.zimage_world_pipeline import ZImageWorldPipeline, ZImageWorldConfig, QUALITY_PRESETS

    # Configure
    if args.quality in QUALITY_PRESETS:
        config = QUALITY_PRESETS[args.quality]
    else:
        config = ZImageWorldConfig(
            height=args.resolution, width=args.resolution,
            num_inference_steps=args.steps, context_frames=args.context,
        )
    config.device = device
    config.temporal_every_n = args.temporal_every_n
    config.checkpoint_path = args.checkpoint
    config.compile_model = args.compile

    # Load pipeline
    print("\n=== Loading Z-Image World Model ===")
    pipeline = ZImageWorldPipeline.from_pretrained(config=config)

    # Generate initial frame
    print("\n=== Generating Initial Frame ===")
    initial_frame = generate_initial_frame(args.prompt, config.height, device)
    pipeline.set_initial_frame(initial_frame)

    # Warmup
    pipeline.warmup(num_iterations=2)

    # Launch pygame demo
    print("\n=== Starting Interactive Demo ===")
    print("Controls:")
    print("  WASD / Arrow keys: Move")
    print("  IJKL: Look around")
    print("  Space: Jump | C: Crouch | E: Interact | F: Attack")
    print("  Q: Cycle quality preset")
    print("  H: Toggle help overlay")
    print("  R: Toggle recording")
    print("  ESC: Quit")

    try:
        import pygame
        pygame.init()

        window_size = (args.window_width, args.window_height)
        screen = pygame.display.set_mode(window_size)
        pygame.display.set_caption("Z-Image World Model Demo")
        clock = pygame.time.Clock()
        font = pygame.font.Font(None, 36)

        from models.action_encoder import ActionSpace

        running = True
        show_help = True
        quality_idx = ["fast", "balanced", "quality"].index(args.quality)
        quality_names = ["fast", "balanced", "quality"]

        while running:
            # Handle events
            keys_pressed = set()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_h:
                        show_help = not show_help
                    elif event.key == pygame.K_q:
                        quality_idx = (quality_idx + 1) % 3
                        pipeline.set_quality(quality_names[quality_idx])

            # Get current key state
            key_state = pygame.key.get_pressed()
            if key_state[pygame.K_w] or key_state[pygame.K_UP]:
                keys_pressed.add("w")
            if key_state[pygame.K_s] or key_state[pygame.K_DOWN]:
                keys_pressed.add("s")
            if key_state[pygame.K_a] or key_state[pygame.K_LEFT]:
                keys_pressed.add("a")
            if key_state[pygame.K_d] or key_state[pygame.K_RIGHT]:
                keys_pressed.add("d")
            if key_state[pygame.K_i]:
                keys_pressed.add("i")
            if key_state[pygame.K_j]:
                keys_pressed.add("j")
            if key_state[pygame.K_k]:
                keys_pressed.add("k")
            if key_state[pygame.K_l]:
                keys_pressed.add("l")
            if key_state[pygame.K_SPACE]:
                keys_pressed.add("space")
            if key_state[pygame.K_c]:
                keys_pressed.add("c")
            if key_state[pygame.K_e]:
                keys_pressed.add("e")
            if key_state[pygame.K_f]:
                keys_pressed.add("f")

            # Convert to action
            action = ActionSpace.from_keyboard(keys_pressed)

            # Generate next frame
            frame = pipeline.step(int(action))

            # Convert tensor to pygame surface
            frame_np = frame[0].float().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
            frame_np = (frame_np * 255).astype(np.uint8)
            surface = pygame.surfarray.make_surface(frame_np.swapaxes(0, 1))
            surface = pygame.transform.scale(surface, window_size)

            # Render
            screen.blit(surface, (0, 0))

            # FPS overlay
            fps_text = font.render(
                f"FPS: {pipeline.fps:.1f} | {quality_names[quality_idx]} | "
                f"{pipeline.config.height}x{pipeline.config.width}",
                True, (255, 255, 0),
            )
            screen.blit(fps_text, (10, 10))

            # Action overlay
            action_text = font.render(f"Action: {ActionSpace(action).name}", True, (255, 255, 255))
            screen.blit(action_text, (10, 50))

            # Help overlay
            if show_help:
                help_lines = [
                    "WASD: Move | IJKL: Look",
                    "Space: Jump | Q: Quality",
                    "H: Help | ESC: Quit",
                ]
                for i, line in enumerate(help_lines):
                    text = font.render(line, True, (200, 200, 200))
                    screen.blit(text, (10, window_size[1] - 120 + i * 35))

            pygame.display.flip()
            clock.tick(30)  # Cap display at 30 FPS

        pygame.quit()

    except ImportError:
        print("\npygame not installed. Running in headless mode (generating 10 frames)...")
        for i in range(10):
            action = 0 if i < 5 else 2  # Forward then left
            frame = pipeline.step(action)
            print(f"  Frame {i + 1}: {frame.shape}, FPS: {pipeline.fps:.1f}")
        print(f"Average FPS: {pipeline.fps:.1f}")


def main():
    parser = argparse.ArgumentParser(description="Z-Image World Model Interactive Demo")
    parser.add_argument("--quality", type=str, default="balanced", choices=["fast", "balanced", "quality"])
    parser.add_argument("--prompt", type=str, default="a colorful landscape with mountains and a river, video game style")
    parser.add_argument("--checkpoint", type=str, default=None, help="Trained checkpoint path")
    parser.add_argument("--temporal_every_n", type=int, default=3, help="Temporal attention frequency")
    parser.add_argument("--resolution", type=int, default=256, help="Generation resolution (custom)")
    parser.add_argument("--steps", type=int, default=2, help="Denoising steps (custom)")
    parser.add_argument("--context", type=int, default=2, help="Context frames (custom)")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile")
    parser.add_argument("--window_width", type=int, default=800, help="Display window width")
    parser.add_argument("--window_height", type=int, default=800, help="Display window height")
    args = parser.parse_args()

    run_demo(args)


if __name__ == "__main__":
    main()
