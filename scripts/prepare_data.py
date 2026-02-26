#!/usr/bin/env python3
"""
Prepare video data for Z-Image World Model training.

Downloads public domain video clips and splits them into training segments.

Usage:
    # Download and prepare all data
    uv run python scripts/prepare_data.py

    # Only generate synthetic videos
    uv run python scripts/prepare_data.py --synthetic-only --num-synthetic 100

    # Custom output directory
    uv run python scripts/prepare_data.py --output-dir data/my_videos
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch


# Public domain / Creative Commons video URLs
VIDEO_SOURCES = [
    # Big Buck Bunny (Blender, CC-BY-3.0)
    ("Big_Buck_Bunny_360_10s_1MB", "https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/360/Big_Buck_Bunny_360_10s_1MB.mp4"),
    ("Big_Buck_Bunny_360_10s_2MB", "https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/360/Big_Buck_Bunny_360_10s_2MB.mp4"),
    # Sintel (Blender, CC-BY-3.0)
    ("Sintel_360_10s_1MB", "https://test-videos.co.uk/vids/sintel/mp4/h264/360/Sintel_360_10s_1MB.mp4"),
    ("Sintel_360_10s_2MB", "https://test-videos.co.uk/vids/sintel/mp4/h264/360/Sintel_360_10s_2MB.mp4"),
    # Jellyfish (public domain, great for motion)
    ("Jellyfish_360_10s_1MB", "https://test-videos.co.uk/vids/jellyfish/mp4/h264/360/Jellyfish_360_10s_1MB.mp4"),
    ("Jellyfish_360_10s_2MB", "https://test-videos.co.uk/vids/jellyfish/mp4/h264/360/Jellyfish_360_10s_2MB.mp4"),
]


def download_videos(output_dir: Path) -> list[Path]:
    """Download public domain video clips."""
    output_dir.mkdir(parents=True, exist_ok=True)
    downloaded = []

    for name, url in VIDEO_SOURCES:
        output_path = output_dir / f"{name}.mp4"
        if output_path.exists():
            print(f"  Already exists: {output_path.name}")
            downloaded.append(output_path)
            continue

        print(f"  Downloading {name}...")
        try:
            import urllib.request
            urllib.request.urlretrieve(url, output_path)
            downloaded.append(output_path)
            print(f"    Saved: {output_path.name}")
        except Exception as e:
            print(f"    Failed: {e}")

    return downloaded


def split_video_to_clips(
    video_path: Path, output_dir: Path, clip_duration: int = 5, overlap: int = 2,
) -> list[Path]:
    """Split a video into overlapping clips using ffmpeg."""
    clips = []

    # Get video duration
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
            capture_output=True, text=True,
        )
        duration = float(result.stdout.strip())
    except (subprocess.CalledProcessError, ValueError):
        print(f"  Could not get duration for {video_path.name}, using as-is")
        return [video_path]

    # Split into clips
    clip_dir = output_dir / "clips"
    clip_dir.mkdir(parents=True, exist_ok=True)
    stem = video_path.stem

    start = 0.0
    clip_idx = 0
    while start + clip_duration <= duration:
        clip_path = clip_dir / f"{stem}_clip{clip_idx:03d}.mp4"
        if not clip_path.exists():
            subprocess.run(
                ["ffmpeg", "-y", "-ss", str(start), "-i", str(video_path),
                 "-t", str(clip_duration), "-c", "copy", "-an", str(clip_path)],
                capture_output=True,
            )
        clips.append(clip_path)
        start += clip_duration - overlap
        clip_idx += 1

    if not clips:
        clips.append(video_path)

    return clips


def generate_synthetic_videos(
    output_dir: Path, num_videos: int = 50, num_frames: int = 30, resolution: int = 256,
) -> list[Path]:
    """Generate synthetic videos with moving patterns."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    try:
        import imageio
    except ImportError:
        print("  imageio not installed, skipping synthetic video generation")
        print("  Install with: uv sync --extra video")
        return []

    patterns = ["circle", "linear", "zoom", "rotate", "wave"]

    for i in range(num_videos):
        path = output_dir / f"synthetic_{i:04d}.mp4"
        if path.exists():
            paths.append(path)
            continue

        pattern = patterns[i % len(patterns)]
        frames = []

        for f in range(num_frames):
            t = f / num_frames
            frame = np.zeros((resolution, resolution, 3), dtype=np.uint8)

            # Background gradient
            bg_r = int(50 + 50 * np.sin(2 * np.pi * t + i * 0.5))
            bg_g = int(80 + 40 * np.cos(2 * np.pi * t + i * 0.3))
            bg_b = int(60 + 60 * np.sin(2 * np.pi * t * 2 + i))
            frame[:, :] = [bg_r, bg_g, bg_b]

            # Moving object
            cx = int(resolution * (0.5 + 0.3 * np.sin(2 * np.pi * t)))
            cy = int(resolution * (0.5 + 0.3 * np.cos(2 * np.pi * t)))
            radius = int(resolution * 0.1)

            y, x = np.ogrid[:resolution, :resolution]
            mask = (x - cx) ** 2 + (y - cy) ** 2 < radius ** 2
            color = [
                max(0, min(255, int(200 + 55 * np.sin(i * 1.1)))),
                max(0, min(255, int(100 + 155 * np.cos(i * 0.7)))),
                max(0, min(255, int(150 + 105 * np.sin(i * 1.3)))),
            ]
            frame[mask] = color

            # Add noise for realism
            noise = np.random.randint(0, 15, frame.shape, dtype=np.uint8)
            frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            frames.append(frame)

        writer = imageio.get_writer(str(path), fps=10)
        for frame in frames:
            writer.append_data(frame)
        writer.close()
        paths.append(path)

    return paths


def main():
    parser = argparse.ArgumentParser(description="Prepare video data for training")
    parser.add_argument("--output-dir", type=str, default="data/videos", help="Output directory")
    parser.add_argument("--synthetic-only", action="store_true", help="Only generate synthetic videos")
    parser.add_argument("--num-synthetic", type=int, default=50, help="Number of synthetic videos")
    parser.add_argument("--clip-duration", type=int, default=5, help="Clip duration in seconds")
    parser.add_argument("--resolution", type=int, default=256, help="Synthetic video resolution")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    all_clips = []

    if not args.synthetic_only:
        # Download real videos
        print("=== Downloading Public Domain Videos ===")
        raw_dir = output_dir / "raw"
        videos = download_videos(raw_dir)
        print(f"Downloaded {len(videos)} videos")

        # Split into clips
        if videos:
            print("\n=== Splitting into Clips ===")
            has_ffmpeg = subprocess.run(
                ["which", "ffmpeg"], capture_output=True,
            ).returncode == 0

            if has_ffmpeg:
                for v in videos:
                    clips = split_video_to_clips(v, output_dir, args.clip_duration)
                    all_clips.extend(clips)
                    print(f"  {v.name}: {len(clips)} clips")
            else:
                print("  ffmpeg not found, using raw videos as-is")
                all_clips.extend(videos)

    # Generate synthetic videos
    print(f"\n=== Generating {args.num_synthetic} Synthetic Videos ===")
    synthetic_dir = output_dir / "synthetic"
    synthetic = generate_synthetic_videos(
        synthetic_dir, args.num_synthetic, resolution=args.resolution,
    )
    all_clips.extend(synthetic)
    print(f"Generated {len(synthetic)} synthetic videos")

    # Summary
    print(f"\n{'=' * 60}")
    print(f"DATA PREPARATION COMPLETE")
    print(f"{'=' * 60}")
    print(f"Total clips: {len(all_clips)}")
    print(f"Output directory: {output_dir}")

    # List all video files for training
    all_videos = sorted(output_dir.rglob("*.mp4"))
    print(f"All video files found: {len(all_videos)}")
    print(f"\nTo train:")
    print(f"  uv run python scripts/train_zimage_world.py --data_dir {output_dir} --epochs 50")


if __name__ == "__main__":
    main()
