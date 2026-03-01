#!/usr/bin/env python3
"""
Generate rich synthetic video data for Stage 1 temporal training.

Produces videos with substantially more temporal structure than the original
simple-circle generator, to push temporal attention gamma from ~0.025 toward
0.1+:

  - Multi-ball physics (gravity, bounce, collision)
  - Camera motion (pan, tilt, zoom over procedural scenes)
  - Animated background textures (wave fields, gradient evolution)
  - Layered scenes (background + midground + foreground objects)
  - Smooth lighting changes (time-of-day cycling)

Usage:
    python scripts/generate_rich_synthetic.py --num_videos 500 --output_dir data/videos/synthetic_rich
    python scripts/generate_rich_synthetic.py --num_videos 200 --output_dir data/videos/synthetic_rich --overwrite
"""

import argparse
import math
import os
from pathlib import Path

import imageio
import numpy as np


RNG = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# Low-level drawing helpers
# ---------------------------------------------------------------------------

def _clamp(arr, lo=0, hi=255):
    return np.clip(arr, lo, hi).astype(np.uint8)


def _draw_circle(frame, cx, cy, r, color, alpha=1.0):
    """Draw a filled circle with optional alpha blending."""
    H, W = frame.shape[:2]
    ys = np.arange(max(0, cy - r - 1), min(H, cy + r + 2))
    xs = np.arange(max(0, cx - r - 1), min(W, cx + r + 2))
    yy, xx = np.meshgrid(ys, xs, indexing='ij')
    dist2 = (yy - cy) ** 2 + (xx - cx) ** 2
    mask = dist2 < r * r
    if mask.any():
        frame[ys[mask.any(axis=1)][:, None], xs[mask.any(axis=0)]] = _blend(
            frame[ys[mask.any(axis=1)][:, None], xs[mask.any(axis=0)]],
            np.array(color, dtype=np.float32),
            alpha,
        )
    # simpler: just draw via boolean mask
    yy2, xx2 = np.ogrid[max(0, cy-r-1):min(H, cy+r+2), max(0, cx-r-1):min(W, cx+r+2)]
    m = (yy2 - cy)**2 + (xx2 - cx)**2 < r*r
    sl = (slice(max(0, cy-r-1), min(H, cy+r+2)),
          slice(max(0, cx-r-1), min(W, cx+r+2)))
    if alpha >= 1.0:
        frame[sl][m] = color
    else:
        frame[sl][m] = (frame[sl][m] * (1 - alpha) + np.array(color) * alpha).astype(np.uint8)


def _blend(a, b, alpha):
    return (a * (1 - alpha) + b * alpha).astype(np.uint8)


def _wave_texture(H, W, t, freq=4.0, speed=1.0, color_a=(30, 60, 120), color_b=(80, 140, 200)):
    """Animated sine-wave background."""
    xs = np.linspace(0, 2 * math.pi * freq, W)
    ys = np.linspace(0, 2 * math.pi * freq, H)
    xx, yy = np.meshgrid(xs, ys)
    val = 0.5 + 0.5 * np.sin(xx + speed * t) * np.cos(yy * 0.7 + speed * t * 1.3)
    ca = np.array(color_a, dtype=np.float32)
    cb = np.array(color_b, dtype=np.float32)
    frame = (ca[None, None] * (1 - val[:, :, None]) + cb[None, None] * val[:, :, None])
    return _clamp(frame)


def _gradient_bg(H, W, t, hue_shift=0.0):
    """Slowly rotating gradient background."""
    xs = np.linspace(-1, 1, W)
    ys = np.linspace(-1, 1, H)
    xx, yy = np.meshgrid(xs, ys)
    angle = math.pi / 4 + hue_shift + t * 0.3
    val = 0.5 + 0.5 * (xx * math.cos(angle) + yy * math.sin(angle))
    # Map to hue
    r = _clamp(40 + 180 * val)
    g = _clamp(80 + 120 * np.roll(val, H // 3, axis=0))
    b = _clamp(60 + 180 * (1 - val))
    return np.stack([r, g, b], axis=2)


def _checkerboard_bg(H, W, t, size=32):
    """Moving checkerboard."""
    shift_x = int(t * 20) % size
    shift_y = int(t * 15) % size
    frame = np.zeros((H, W, 3), dtype=np.uint8)
    for y in range(H):
        for x in range(W):
            bx = ((x + shift_x) // size) % 2
            by = ((y + shift_y) // size) % 2
            if bx == by:
                frame[y, x] = [40, 40, 60]
            else:
                frame[y, x] = [200, 180, 160]
    return frame


def _perlin_like_bg(H, W, t, seed=0):
    """Pseudo-Perlin animated background using summed sinusoids."""
    frame = np.zeros((H, W, 3), dtype=np.float32)
    rng = np.random.default_rng(seed)
    for _ in range(6):
        fx = rng.uniform(1, 8)
        fy = rng.uniform(1, 8)
        ft = rng.uniform(0.5, 3.0)
        phase = rng.uniform(0, 2 * math.pi)
        amp = rng.uniform(20, 60)
        ch = rng.integers(0, 3)
        xs = np.linspace(0, 2 * math.pi * fx, W)
        ys = np.linspace(0, 2 * math.pi * fy, H)
        xx, yy = np.meshgrid(xs, ys)
        frame[:, :, ch] += amp * np.sin(xx + yy * 0.5 + ft * t + phase)

    base = np.array([80, 100, 120], dtype=np.float32)
    frame += base[None, None]
    return _clamp(frame)


# ---------------------------------------------------------------------------
# Scene generators
# ---------------------------------------------------------------------------

def scene_physics_balls(H, W, num_frames, rng):
    """Multiple balls bouncing with gravity and wall reflection."""
    num_balls = rng.integers(3, 9)
    radii = rng.integers(int(H * 0.04), int(H * 0.12), size=num_balls)
    pos = np.stack([
        rng.uniform(radii, W - radii),
        rng.uniform(radii, H - radii),
    ], axis=1).astype(np.float32)
    vel = rng.uniform(-3, 3, size=(num_balls, 2)).astype(np.float32)
    colors = rng.integers(80, 256, size=(num_balls, 3))
    bg_seed = rng.integers(0, 10000)
    gravity = 0.15

    frames = []
    for fi in range(num_frames):
        t = fi / num_frames
        bg = _perlin_like_bg(H, W, t * 2, seed=bg_seed)
        frame = bg.copy()

        # Physics update
        vel[:, 1] += gravity
        pos += vel

        for i in range(num_balls):
            r = radii[i]
            if pos[i, 0] - r < 0:
                pos[i, 0] = r; vel[i, 0] = abs(vel[i, 0]) * 0.9
            if pos[i, 0] + r > W:
                pos[i, 0] = W - r; vel[i, 0] = -abs(vel[i, 0]) * 0.9
            if pos[i, 1] - r < 0:
                pos[i, 1] = r; vel[i, 1] = abs(vel[i, 1]) * 0.9
            if pos[i, 1] + r > H:
                pos[i, 1] = H - r; vel[i, 1] = -abs(vel[i, 1]) * 0.85

            cx, cy = int(pos[i, 0]), int(pos[i, 1])
            _draw_circle(frame, cx, cy, r, colors[i].tolist())

        # Shadow / glow (lighter version behind ball)
        frames.append(frame)

    return frames


def scene_camera_zoom(H, W, num_frames, rng):
    """Zoom into/out of a procedural scene."""
    bg_seed = rng.integers(0, 10000)
    bg = _perlin_like_bg(H * 2, W * 2, 0.0, seed=bg_seed)

    num_objs = rng.integers(3, 7)
    obj_pos = rng.uniform(0.1, 1.9, size=(num_objs, 2))  # in [0,2]×[0,2] of 2H×2W bg
    obj_colors = rng.integers(100, 256, size=(num_objs, 3))
    obj_radii = rng.integers(int(H * 0.05), int(H * 0.18), size=num_objs)

    # Draw objects onto background
    scene = bg.copy()
    for i in range(num_objs):
        cx = int(obj_pos[i, 0] * W)
        cy = int(obj_pos[i, 1] * H)
        _draw_circle(scene, cx, cy, obj_radii[i], obj_colors[i].tolist())

    zoom_start = rng.uniform(0.4, 0.6)
    zoom_end = rng.uniform(0.7, 1.2)
    pan_x = rng.integers(0, W // 2)
    pan_y = rng.integers(0, H // 2)

    frames = []
    for fi in range(num_frames):
        alpha = fi / max(num_frames - 1, 1)
        zoom = zoom_start + (zoom_end - zoom_start) * alpha
        crop_w = int(W * zoom)
        crop_h = int(H * zoom)
        sx = int(pan_x * alpha + (2 * W - crop_w) // 4)
        sy = int(pan_y * alpha + (2 * H - crop_h) // 4)
        sx = max(0, min(2 * W - crop_w, sx))
        sy = max(0, min(2 * H - crop_h, sy))
        crop = scene[sy:sy + crop_h, sx:sx + crop_w]
        from PIL import Image as PILImage
        pil = PILImage.fromarray(crop).resize((W, H), PILImage.BILINEAR)
        frames.append(np.array(pil))

    return frames


def scene_camera_pan(H, W, num_frames, rng):
    """Pan across a wide procedural scene."""
    bg_w = W * 3
    bg_seed = rng.integers(0, 10000)
    bg = _perlin_like_bg(H, bg_w, 0.0, seed=bg_seed)

    num_objs = rng.integers(5, 12)
    obj_colors = rng.integers(80, 256, size=(num_objs, 3))
    obj_radii = rng.integers(int(H * 0.04), int(H * 0.15), size=num_objs)
    for i in range(num_objs):
        cx = rng.integers(obj_radii[i], bg_w - obj_radii[i])
        cy = rng.integers(obj_radii[i], H - obj_radii[i])
        _draw_circle(bg, cx, cy, obj_radii[i], obj_colors[i].tolist())

    direction = rng.choice([-1, 1])
    start_x = rng.integers(0, bg_w - W) if direction == 1 else rng.integers(W, bg_w)

    frames = []
    for fi in range(num_frames):
        alpha = fi / max(num_frames - 1, 1)
        speed = (bg_w - W) * 0.5
        sx = int(start_x + direction * speed * alpha)
        sx = max(0, min(bg_w - W, sx))
        frames.append(bg[:, sx:sx + W].copy())

    return frames


def scene_wave_objects(H, W, num_frames, rng):
    """Wave-animated background with orbiting objects."""
    color_a = tuple(rng.integers(20, 100, size=3).tolist())
    color_b = tuple(rng.integers(100, 240, size=3).tolist())
    freq = rng.uniform(2, 6)
    speed = rng.uniform(0.5, 2.5)

    num_orbiters = rng.integers(2, 6)
    orbit_r = [rng.uniform(H * 0.15, H * 0.4) for _ in range(num_orbiters)]
    orbit_speed = [rng.uniform(0.5, 3.0) * rng.choice([-1, 1]) for _ in range(num_orbiters)]
    orbit_phase = [rng.uniform(0, 2 * math.pi) for _ in range(num_orbiters)]
    obj_radii = rng.integers(int(H * 0.04), int(H * 0.1), size=num_orbiters)
    obj_colors = rng.integers(150, 256, size=(num_orbiters, 3))

    frames = []
    for fi in range(num_frames):
        t = fi * 0.15
        frame = _wave_texture(H, W, t, freq=freq, speed=speed,
                              color_a=color_a, color_b=color_b)

        for i in range(num_orbiters):
            angle = orbit_phase[i] + orbit_speed[i] * t
            cx = int(W / 2 + orbit_r[i] * math.cos(angle))
            cy = int(H / 2 + orbit_r[i] * math.sin(angle))
            _draw_circle(frame, cx, cy, int(obj_radii[i]), obj_colors[i].tolist())

        frames.append(frame)

    return frames


def scene_gradient_flow(H, W, num_frames, rng):
    """Gradient background with objects floating across."""
    hue_shift = rng.uniform(0, 2 * math.pi)
    num_objects = rng.integers(3, 8)
    shapes = rng.choice(['circle', 'square'], size=num_objects)
    obj_colors = rng.integers(100, 256, size=(num_objects, 3))
    obj_sizes = rng.integers(int(H * 0.04), int(H * 0.14), size=num_objects)
    # Start positions (can be off-screen)
    start_x = rng.uniform(-0.2, 1.2, size=num_objects) * W
    start_y = rng.uniform(0.1, 0.9, size=num_objects) * H
    vel_x = rng.uniform(-2.5, 2.5, size=num_objects)
    vel_y = rng.uniform(-1.5, 1.5, size=num_objects)

    frames = []
    for fi in range(num_frames):
        t = fi * 0.12
        frame = _gradient_bg(H, W, t, hue_shift)

        for i in range(num_objects):
            cx = int(start_x[i] + vel_x[i] * fi)
            cy = int(start_y[i] + vel_y[i] * fi)
            r = obj_sizes[i]
            if shapes[i] == 'circle':
                _draw_circle(frame, cx, cy, r, obj_colors[i].tolist())
            else:
                # Square
                sy = max(0, cy - r); ey = min(H, cy + r)
                sx = max(0, cx - r); ex = min(W, cx + r)
                frame[sy:ey, sx:ex] = obj_colors[i]

        frames.append(frame)

    return frames


def scene_layered_parallax(H, W, num_frames, rng):
    """Parallax scrolling with background / midground / foreground layers."""
    from PIL import Image as PILImage

    # Background: slow scroll
    bg_w = W * 4
    bg = np.zeros((H, bg_w, 3), dtype=np.uint8)
    # Sky gradient
    sky_top = rng.integers(30, 100, size=3)
    sky_bot = rng.integers(100, 200, size=3)
    for y in range(H // 2):
        alpha = y / (H / 2)
        bg[y, :] = (sky_top * (1 - alpha) + sky_bot * alpha).astype(np.uint8)
    # Ground
    ground_col = rng.integers(40, 120, size=3)
    bg[H // 2:, :] = ground_col
    # Scattered objects in background (trees, rocks)
    num_bg_objs = rng.integers(10, 25)
    for _ in range(num_bg_objs):
        cx = rng.integers(0, bg_w)
        cy = rng.integers(H // 4, H)
        r = rng.integers(int(H * 0.03), int(H * 0.1))
        col = rng.integers(50, 180, size=3).tolist()
        _draw_circle(bg, cx, cy, r, col)

    # Foreground: fast scroll, larger objects
    fg_w = W * 2
    fg = np.zeros((H, fg_w, 4), dtype=np.uint8)  # RGBA
    num_fg_objs = rng.integers(3, 8)
    for _ in range(num_fg_objs):
        cx = rng.integers(0, fg_w)
        cy = rng.integers(H // 2, H)
        r = rng.integers(int(H * 0.08), int(H * 0.2))
        col = rng.integers(100, 220, size=3).tolist() + [200]
        # Just draw on fg with alpha
        _draw_circle(fg[:, :, :3], cx, cy, r, col[:3])
        _draw_circle(fg[:, :, 3:4].squeeze(2), cx, cy, r, [200])

    pan_speed_bg = bg_w / num_frames * 0.5
    pan_speed_fg = fg_w / num_frames * 0.8

    frames = []
    for fi in range(num_frames):
        bx = int(fi * pan_speed_bg) % (bg_w - W)
        frame = bg[:, bx:bx + W].copy()

        # Composite foreground
        fx = int(fi * pan_speed_fg) % (fg_w - W)
        fg_crop = fg[:, fx:fx + W]
        alpha_mask = fg_crop[:, :, 3:4] / 255.0
        frame = (frame * (1 - alpha_mask) + fg_crop[:, :, :3] * alpha_mask).astype(np.uint8)

        frames.append(frame)

    return frames


def scene_growing_objects(H, W, num_frames, rng):
    """Objects that grow, shrink, appear and disappear."""
    bg_seed = rng.integers(0, 10000)
    num_objects = rng.integers(4, 10)
    life_starts = rng.integers(0, num_frames // 2, size=num_objects)
    life_durations = rng.integers(num_frames // 3, num_frames, size=num_objects)
    max_radii = rng.integers(int(H * 0.05), int(H * 0.2), size=num_objects)
    centers_x = rng.integers(int(H * 0.1), W - int(H * 0.1), size=num_objects)
    centers_y = rng.integers(int(H * 0.1), H - int(H * 0.1), size=num_objects)
    colors = rng.integers(80, 256, size=(num_objects, 3))

    frames = []
    for fi in range(num_frames):
        t = fi / num_frames
        frame = _perlin_like_bg(H, W, t, seed=bg_seed)
        for i in range(num_objects):
            age = fi - life_starts[i]
            if age < 0 or age >= life_durations[i]:
                continue
            life_frac = age / life_durations[i]
            # Grow then shrink (bell curve)
            size_frac = math.sin(math.pi * life_frac)
            r = int(max_radii[i] * size_frac)
            if r < 2:
                continue
            _draw_circle(frame, centers_x[i], centers_y[i], r, colors[i].tolist())
        frames.append(frame)
    return frames


SCENES = [
    scene_physics_balls,
    scene_camera_zoom,
    scene_camera_pan,
    scene_wave_objects,
    scene_gradient_flow,
    scene_layered_parallax,
    scene_growing_objects,
]


def generate_video(output_path: Path, rng, resolution: int = 256, num_frames: int = 32, fps: int = 10):
    """Generate a single rich synthetic video."""
    H = W = resolution
    scene_fn = rng.choice(SCENES)

    try:
        frames = scene_fn(H, W, num_frames, rng)
    except Exception as e:
        # Fallback to basic physics if a scene fails
        frames = scene_physics_balls(H, W, num_frames, rng)

    # Add small amount of noise for realism
    noise_level = rng.integers(3, 12)
    noisy = []
    for f in frames:
        noise = rng.integers(0, noise_level, f.shape, dtype=np.uint8)
        noisy.append(np.clip(f.astype(np.int16) + noise, 0, 255).astype(np.uint8))

    with imageio.get_writer(str(output_path), fps=fps, macro_block_size=None) as writer:
        for frame in noisy:
            writer.append_data(frame)


def main():
    parser = argparse.ArgumentParser(description="Generate rich synthetic video training data")
    parser.add_argument("--num_videos", type=int, default=500)
    parser.add_argument("--output_dir", type=str, default="data/videos/synthetic_rich")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--num_frames", type=int, default=32)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    generated = 0
    skipped = 0
    errors = 0

    print(f"Generating {args.num_videos} rich synthetic videos → {out_dir}")
    print(f"  Resolution: {args.resolution}×{args.resolution}, {args.num_frames} frames @ {args.fps} fps")
    print(f"  Scenes: {[s.__name__ for s in SCENES]}")
    print()

    for i in range(args.num_videos):
        path = out_dir / f"rich_{i:04d}.mp4"
        if path.exists() and not args.overwrite:
            skipped += 1
            continue
        try:
            generate_video(path, rng, resolution=args.resolution,
                           num_frames=args.num_frames, fps=args.fps)
            generated += 1
            if (i + 1) % 50 == 0:
                print(f"  [{i+1}/{args.num_videos}] Generated {generated} videos")
        except Exception as e:
            print(f"  [{i+1}] ERROR: {e}")
            errors += 1

    print(f"\nDone: {generated} generated, {skipped} skipped, {errors} errors")
    print(f"Output: {out_dir}")
    print(f"\nTrain command:")
    print(f"  python scripts/train_zimage_world.py \\")
    print(f"    --model_path weights/Z-Image-Turbo \\")
    print(f"    --stage1_checkpoint checkpoints/zimage_stage1_v2/world_model_final.pt \\")
    print(f"    --data_dir {out_dir} \\")
    print(f"    --epochs 30 --lr 5e-5 --num_frames 4 --resolution {args.resolution}")


if __name__ == "__main__":
    main()
