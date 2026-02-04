#!/usr/bin/env python3
"""
Z-Image World Model Demo

Uses Z-Image for image-to-image transformation based on actions.
This is a simplified demo that shows beautiful output.
"""

import argparse
import time
from pathlib import Path

import torch
import numpy as np
from PIL import Image

# Action prompts for different movements
ACTION_PROMPTS = {
    "idle": "same scene, no change",
    "forward": "same scene from a slightly closer perspective, moving forward",
    "backward": "same scene from a slightly farther perspective, moving backward",
    "left": "same scene shifted slightly to the right, camera moving left",
    "right": "same scene shifted slightly to the left, camera moving right",
    "look_up": "same scene with camera tilted slightly upward",
    "look_down": "same scene with camera tilted slightly downward",
    "jump": "same scene from a slightly higher viewpoint",
}


def load_pipeline(device: str = "cuda"):
    """Load Z-Image Img2Img pipeline."""
    from diffusers import ZImageImg2ImgPipeline

    print("Loading Z-Image-Turbo model...")
    pipe = ZImageImg2ImgPipeline.from_pretrained(
        "Tongyi-MAI/Z-Image-Turbo",
        torch_dtype=torch.bfloat16,
    )
    pipe.to(device)
    print("Model loaded!")
    return pipe


def generate_initial_image(pipe, prompt: str, size: tuple = (512, 512)) -> Image.Image:
    """Generate an initial image from text prompt."""
    from diffusers import ZImagePipeline

    # Use text-to-image for initial generation
    t2i_pipe = ZImagePipeline.from_pretrained(
        "Tongyi-MAI/Z-Image-Turbo",
        torch_dtype=torch.bfloat16,
    )
    t2i_pipe.to(pipe.device)

    print(f"Generating initial image: '{prompt}'")
    image = t2i_pipe(
        prompt=prompt,
        height=size[0],
        width=size[1],
        num_inference_steps=9,
        guidance_scale=0.0,
        generator=torch.Generator(pipe.device).manual_seed(42),
    ).images[0]

    # Clean up to save memory
    del t2i_pipe
    torch.cuda.empty_cache()

    return image


def step(pipe, image: Image.Image, action: str, strength: float = 0.3) -> Image.Image:
    """Generate next frame based on action.

    Args:
        pipe: Z-Image Img2Img pipeline
        image: Current frame
        action: Action name (forward, backward, left, right, etc.)
        strength: How much to transform (0-1, lower = less change)

    Returns:
        Next frame
    """
    prompt = ACTION_PROMPTS.get(action, ACTION_PROMPTS["idle"])

    # Ensure image is the right size
    if image.size != (512, 512):
        image = image.resize((512, 512), Image.Resampling.LANCZOS)

    # Generate next frame
    next_image = pipe(
        prompt=prompt,
        image=image,
        strength=strength,
        num_inference_steps=9,
        guidance_scale=0.0,
        generator=torch.Generator(pipe.device).manual_seed(int(time.time() * 1000) % 2**32),
    ).images[0]

    return next_image


def run_demo(
    output_dir: str = "demo_output",
    num_frames: int = 30,
    initial_prompt: str = "A beautiful fantasy landscape with mountains, a crystal lake, and a medieval castle in the distance, sunset lighting, highly detailed",
):
    """Run the demo and save frames."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Load pipeline
    pipe = load_pipeline()

    # Generate initial image
    current_image = generate_initial_image(pipe, initial_prompt)
    current_image.save(output_path / "frame_000.png")
    print(f"Saved initial frame to {output_path / 'frame_000.png'}")

    # Define action sequence (simulate walking forward and looking around)
    actions = [
        "forward", "forward", "forward",
        "look_left", "look_left",
        "forward", "forward",
        "look_right", "look_right", "look_right",
        "forward", "forward", "forward",
        "look_up",
        "forward", "forward",
        "look_down", "look_down",
        "forward", "forward", "forward",
        "left", "left",
        "forward", "forward",
        "right", "right", "right",
        "forward", "forward", "jump",
    ]

    # Generate frames
    print(f"\nGenerating {num_frames} frames...")
    times = []

    for i in range(min(num_frames, len(actions))):
        action = actions[i]
        print(f"Frame {i+1}/{num_frames}: action='{action}'", end=" ")

        start = time.time()
        current_image = step(pipe, current_image, action, strength=0.25)
        elapsed = time.time() - start
        times.append(elapsed)

        # Save frame
        frame_path = output_path / f"frame_{i+1:03d}.png"
        current_image.save(frame_path)
        print(f"({elapsed:.2f}s)")

    # Print stats
    avg_time = np.mean(times)
    fps = 1.0 / avg_time
    print(f"\n{'='*50}")
    print(f"Done! Frames saved to {output_path}/")
    print(f"Average time per frame: {avg_time:.2f}s")
    print(f"Effective FPS: {fps:.2f}")
    print(f"{'='*50}")

    # Create video if ffmpeg available
    try:
        import subprocess
        video_path = output_path / "demo.mp4"
        cmd = [
            "ffmpeg", "-y", "-framerate", "10",
            "-i", str(output_path / "frame_%03d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            str(video_path)
        ]
        subprocess.run(cmd, capture_output=True)
        print(f"Video saved to {video_path}")
    except Exception as e:
        print(f"Could not create video (ffmpeg may not be installed): {e}")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Z-Image World Model Demo")
    parser.add_argument("--output", type=str, default="demo_output", help="Output directory")
    parser.add_argument("--num-frames", type=int, default=30, help="Number of frames to generate")
    parser.add_argument("--prompt", type=str,
                       default="A beautiful fantasy landscape with mountains, a crystal lake, and a medieval castle in the distance, sunset lighting, highly detailed",
                       help="Initial scene description")
    args = parser.parse_args()

    run_demo(
        output_dir=args.output,
        num_frames=args.num_frames,
        initial_prompt=args.prompt,
    )


if __name__ == "__main__":
    main()
