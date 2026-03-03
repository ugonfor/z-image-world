#!/usr/bin/env python3
"""
Generate synthetic action-conditioned video data for Stage 2 training.

Creates video sequences where scene motion is driven by action labels,
producing mp4 + _actions.json pairs ready for scripts/train_zimage_stage2.py.

Each video consists of segments of 4-12 frames per action transition:
  FORWARD (1): camera zoom-in / forward scroll
  BACKWARD (2): camera zoom-out / backward scroll
  LEFT (3):    camera pan left
  RIGHT (4):   camera pan right
  RUN (5):     fast forward motion
  JUMP (6):    vertical bounce (up then down)
  INTERACT (7): flash effect + camera micro-shake
  IDLE (0):    stationary scene with slight noise

Scene types (procedurally generated):
  - corridor:   FPS-style repeating room with pillars and floor/ceiling
  - dungeon:    Dark stones room, torches, archways
  - outdoor:    Landscape with horizon, trees, sky gradient
  - platformer: Side-scrolling with platforms and background layers

Usage:
    # Generate 500 clips (~5 fps, 40 frames each)
    python scripts/generate_action_synthetic.py \\
        --num_videos 500 --output_dir data/videos/stage2_synthetic --fps 10

    # Quick test (20 clips)
    python scripts/generate_action_synthetic.py --num_videos 20 --quick
"""

import argparse
import json
import math
import os
import random
from pathlib import Path

import imageio
import numpy as np


# Action indices must match train_zimage_stage2.py
ACTION_IDLE     = 0
ACTION_FORWARD  = 1
ACTION_BACKWARD = 2
ACTION_LEFT     = 3
ACTION_RIGHT    = 4
ACTION_RUN      = 5
ACTION_JUMP     = 6
ACTION_INTERACT = 7

ACTION_NAMES = ["idle", "forward", "backward", "left", "right", "run", "jump", "interact"]

RNG = np.random.default_rng(None)  # fresh seed each run


# ── Low-level drawing ──────────────────────────────────────────────

def _clamp(arr, lo=0, hi=255):
    return np.clip(arr, lo, hi).astype(np.uint8)


def _lerp(a, b, t):
    return a + (b - a) * t


def _hsv_to_rgb(h, s, v):
    """h in [0,1], s in [0,1], v in [0,1] → (r, g, b) in [0,255]."""
    h6 = h * 6.0
    i = int(h6) % 6
    f = h6 - int(h6)
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    r, g, b = [(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)][i]
    return int(r * 255), int(g * 255), int(b * 255)


def _make_gradient_bg(H, W, col1, col2, angle_deg=90):
    """Create a smooth gradient background."""
    frame = np.zeros((H, W, 3), dtype=np.float32)
    angle = math.radians(angle_deg)
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    ys = np.linspace(-1, 1, H)
    xs = np.linspace(-1, 1, W)
    xx, yy = np.meshgrid(xs, ys)
    t = (xx * cos_a + yy * sin_a + 1) / 2.0
    t = np.clip(t, 0, 1)[..., None]
    frame = (1 - t) * np.array(col1, dtype=np.float32) + t * np.array(col2, dtype=np.float32)
    return _clamp(frame)


def _noise_texture(H, W, scale=32, seed=0):
    """Simple coherent noise texture via downscaled random array."""
    rng = np.random.default_rng(seed)
    small = rng.random((H // scale + 2, W // scale + 2)).astype(np.float32)
    # Nearest-neighbour upsample
    large = np.repeat(np.repeat(small, scale, axis=0), scale, axis=1)
    return large[:H, :W]


# ── Scene state machine ────────────────────────────────────────────

class Scene:
    """Abstract scrolling scene. Subclasses implement render()."""

    def __init__(self, H: int, W: int, seed: int):
        self.H = H
        self.W = W
        self.rng = np.random.default_rng(seed)
        # Camera position (world units)
        self.cam_x = 0.0
        self.cam_y = 0.0
        self.cam_z = 0.0   # depth / zoom
        self.jump_vel = 0.0

    def apply_action(self, action: int, dt: float = 1.0):
        """Move camera according to action."""
        speed = 0.04 * dt
        if action == ACTION_FORWARD:
            self.cam_z += speed * 1.0
        elif action == ACTION_BACKWARD:
            self.cam_z -= speed * 0.8
        elif action == ACTION_LEFT:
            self.cam_x -= speed * 0.9
        elif action == ACTION_RIGHT:
            self.cam_x += speed * 0.9
        elif action == ACTION_RUN:
            self.cam_z += speed * 2.2
        elif action == ACTION_JUMP:
            if self.jump_vel == 0.0:
                self.jump_vel = 0.08
        elif action == ACTION_INTERACT:
            # Micro-shake
            self.cam_x += self.rng.uniform(-0.01, 0.01)
            self.cam_y += self.rng.uniform(-0.005, 0.005)
        # Gravity / jump physics
        if self.jump_vel != 0.0:
            self.cam_y += self.jump_vel * dt
            self.jump_vel -= 0.015 * dt
            if self.cam_y <= 0.0:
                self.cam_y = 0.0
                self.jump_vel = 0.0

    def render(self) -> np.ndarray:
        raise NotImplementedError


# ── Scene implementations ──────────────────────────────────────────

class CorridorScene(Scene):
    """FPS-style hallway that stretches into the distance."""

    def __init__(self, H, W, seed):
        super().__init__(H, W, seed)
        # Palette
        h = self.rng.uniform(0, 1)
        self.floor_col = np.array(_hsv_to_rgb(h, 0.4, 0.4), dtype=np.float32)
        self.ceil_col  = np.array(_hsv_to_rgb(h, 0.3, 0.7), dtype=np.float32)
        self.wall_col  = np.array(_hsv_to_rgb((h + 0.05) % 1, 0.3, 0.55), dtype=np.float32)
        self.pillar_col = np.array(_hsv_to_rgb((h + 0.5) % 1, 0.2, 0.3), dtype=np.float32)
        self.n_pillars = self.rng.integers(3, 8)
        self.cam_z = self.rng.uniform(0.0, 1.0)

    def render(self) -> np.ndarray:
        H, W = self.H, self.W
        frame = np.zeros((H, W, 3), dtype=np.float32)

        horizon = int(H * (0.5 - self.cam_y * 0.3))
        horizon = max(H // 4, min(3 * H // 4, horizon))

        # Ceiling
        frame[:horizon] = self.ceil_col
        # Floor
        frame[horizon:] = self.floor_col

        # Walls (left and right bands)
        cx_pix = int(W * (0.5 + self.cam_x * 0.4))
        wall_w = max(2, int(W * 0.22))
        left_edge  = max(0, cx_pix - wall_w)
        right_edge = min(W, cx_pix + wall_w)
        frame[:, :left_edge]  = self.wall_col
        frame[:, right_edge:] = self.wall_col

        # Receding floor lines (perspective grid)
        z_offset = (self.cam_z % 0.25) / 0.25  # 0..1 repeat
        n_lines = 6
        for i in range(n_lines):
            t = (i / n_lines + z_offset) % 1.0
            depth = t
            if depth < 0.01:
                continue
            y_line = int(horizon + (H - horizon) * depth)
            if 0 < y_line < H:
                frame[y_line, left_edge:right_edge] = self.floor_col * 0.5

        # Ceiling lines
        for i in range(n_lines):
            t = (i / n_lines + z_offset) % 1.0
            depth = t
            y_line = int(horizon - horizon * depth)
            if 0 < y_line < horizon:
                frame[y_line, left_edge:right_edge] = self.ceil_col * 0.6

        # Pillars
        for k in range(self.n_pillars):
            depth_t = ((k / self.n_pillars + self.cam_z * 0.4) % 1.0)
            scale = 0.05 + depth_t * 0.25
            pw = max(2, int(W * scale * 0.12))
            ph_top = int(horizon * (1 - scale * 0.8))
            ph_bot = int(horizon + (H - horizon) * scale * 0.8)
            px = int(left_edge + (right_edge - left_edge) * ((k + 0.5) / self.n_pillars + self.cam_x * 0.1) % 1.0)
            px = max(left_edge, min(right_edge, px))
            shade = max(0, 1.0 - depth_t * 0.7)
            col = (self.pillar_col * shade).astype(np.float32)
            frame[ph_top:ph_bot, max(0, px - pw):min(W, px + pw)] = col

        # Interact flash
        # (handled by caller as overlay)

        return _clamp(frame)


class OutdoorScene(Scene):
    """Open landscape with sky, hills, and trees."""

    def __init__(self, H, W, seed):
        super().__init__(H, W, seed)
        sky_h = self.rng.uniform(0.5, 0.7)
        self.sky_top   = np.array(_hsv_to_rgb(sky_h, 0.6, 0.85), dtype=np.float32)
        self.sky_bot   = np.array(_hsv_to_rgb(sky_h, 0.3, 0.95), dtype=np.float32)
        ground_h = self.rng.uniform(0.25, 0.45)
        self.ground_col = np.array(_hsv_to_rgb(ground_h, 0.5, 0.45), dtype=np.float32)
        self.hill_col   = np.array(_hsv_to_rgb(ground_h, 0.6, 0.35), dtype=np.float32)
        # Tree positions (x offset, height scale)
        n_trees = self.rng.integers(5, 15)
        self.trees = [(self.rng.uniform(0, 1), self.rng.uniform(0.5, 1.5)) for _ in range(n_trees)]
        tree_h = self.rng.uniform(0.25, 0.45)
        self.tree_col   = np.array(_hsv_to_rgb(tree_h, 0.7, 0.3), dtype=np.float32)
        # Horizon level
        self.horizon_frac = self.rng.uniform(0.35, 0.55)
        self.cam_z = self.rng.uniform(0, 2.0)

    def render(self) -> np.ndarray:
        H, W = self.H, self.W
        frame = np.zeros((H, W, 3), dtype=np.float32)

        horizon_y = int(H * (self.horizon_frac - self.cam_y * 0.25))
        horizon_y = max(H // 6, min(5 * H // 6, horizon_y))

        # Sky gradient
        for y in range(horizon_y):
            t = y / max(1, horizon_y)
            frame[y] = _lerp(self.sky_top, self.sky_bot, t)

        # Ground
        frame[horizon_y:] = self.ground_col

        # Hills (sinusoidal)
        scroll = self.cam_x * W * 0.3 + self.cam_z * W * 0.05
        for x in range(W):
            hill_h = (
                math.sin((x + scroll) * 0.03) * 0.06
                + math.sin((x + scroll) * 0.017) * 0.04
            )
            hill_y = int(horizon_y + hill_h * H)
            hill_y = max(0, min(H - 1, hill_y))
            if hill_y < H:
                frame[horizon_y:hill_y, x] = self.hill_col
                if hill_y > horizon_y:
                    frame[horizon_y:hill_y, x] = self.hill_col

        # Trees
        for (tx_frac, t_scale) in self.trees:
            tx = int((tx_frac - self.cam_x * 0.15 + self.cam_z * 0.03) % 1.0 * W)
            tree_w = max(4, int(W * 0.025 * t_scale))
            tree_h_pix = int(H * 0.15 * t_scale)
            ty_bot = horizon_y + int(H * 0.04 * t_scale)
            ty_top = max(0, ty_bot - tree_h_pix)
            x0 = max(0, tx - tree_w)
            x1 = min(W, tx + tree_w)
            frame[ty_top:ty_bot, x0:x1] = self.tree_col

        return _clamp(frame)


class PlatformerScene(Scene):
    """Side-scrolling 2D platformer view."""

    def __init__(self, H, W, seed):
        super().__init__(H, W, seed)
        bg_h = self.rng.uniform(0, 1)
        self.bg_col1 = np.array(_hsv_to_rgb(bg_h, 0.5, 0.8), dtype=np.float32)
        self.bg_col2 = np.array(_hsv_to_rgb((bg_h + 0.1) % 1, 0.4, 0.6), dtype=np.float32)
        plat_h = self.rng.uniform(0.1, 0.4)
        self.plat_col = np.array(_hsv_to_rgb(plat_h, 0.6, 0.5), dtype=np.float32)
        # Platform positions: (x_frac, y_frac, width_frac)
        n_plat = self.rng.integers(3, 8)
        self.platforms = [
            (self.rng.uniform(0, 1.5), self.rng.uniform(0.4, 0.85), self.rng.uniform(0.08, 0.25))
            for _ in range(n_plat)
        ]
        # Ground
        self.ground_frac = self.rng.uniform(0.80, 0.92)
        # Player color
        play_h = self.rng.uniform(0, 1)
        self.player_col = np.array(_hsv_to_rgb(play_h, 0.8, 0.9), dtype=np.float32)
        self.cam_z = 0.0

    def render(self) -> np.ndarray:
        H, W = self.H, self.W
        frame = _make_gradient_bg(H, W, self.bg_col1, self.bg_col2, 180).astype(np.float32)

        # Ground
        gy = int(H * (self.ground_frac - self.cam_y * 0.2))
        gy = max(1, min(H - 1, gy))
        frame[gy:] = self.plat_col * 0.8

        # Platforms
        scroll = self.cam_x * W * 0.5
        for (px_frac, py_frac, pw_frac) in self.platforms:
            px = int((px_frac * W - scroll) % (W * 1.5) - W * 0.25)
            py = int(H * (py_frac - self.cam_y * 0.15))
            pw = max(10, int(W * pw_frac))
            ph = max(4, int(H * 0.04))
            py = max(0, min(H - ph, py))
            x0, x1 = max(0, px), min(W, px + pw)
            if x1 > x0:
                frame[py:py + ph, x0:x1] = self.plat_col

        # Player (fixed center-left)
        player_x = int(W * 0.25)
        player_y = int(H * (self.ground_frac - self.cam_y * 0.2)) - int(H * 0.06)
        player_y = max(0, min(H - 1, player_y))
        pw2, ph2 = max(4, int(W * 0.025)), max(8, int(H * 0.06))
        x0 = max(0, player_x - pw2)
        x1 = min(W, player_x + pw2)
        y0 = max(0, player_y - ph2)
        y1 = min(H, player_y)
        frame[y0:y1, x0:x1] = self.player_col

        return _clamp(frame)


class DungeonScene(Scene):
    """Top-down dungeon room view."""

    def __init__(self, H, W, seed):
        super().__init__(H, W, seed)
        floor_h = self.rng.uniform(0.05, 0.15)
        self.floor_col  = np.array(_hsv_to_rgb(floor_h, 0.3, 0.45), dtype=np.float32)
        self.wall_col   = np.array(_hsv_to_rgb(floor_h, 0.2, 0.25), dtype=np.float32)
        self.torch_col  = np.array([255, 180, 60], dtype=np.float32)
        # Wall offsets
        self.wall_margin = self.rng.uniform(0.08, 0.15)
        # Torch positions
        n_torches = self.rng.integers(2, 6)
        self.torches = [(self.rng.uniform(0.1, 0.9), self.rng.uniform(0.1, 0.9))
                        for _ in range(n_torches)]
        self.cam_z = 0.0

    def render(self) -> np.ndarray:
        H, W = self.H, self.W
        frame = np.zeros((H, W, 3), dtype=np.float32)
        frame[:] = self.floor_col

        # Walls
        m = self.wall_margin
        wm = int(H * m)
        frame[:wm] = self.wall_col
        frame[-wm:] = self.wall_col
        frame[:, :int(W * m)] = self.wall_col
        frame[:, -int(W * m):] = self.wall_col

        # Tile lines (floor texture)
        tile_size = max(4, int(W * 0.08))
        offset_x = int(self.cam_x * W * 0.3) % tile_size
        offset_y = int(self.cam_y * H * 0.3) % tile_size
        for x in range(offset_x, W, tile_size):
            frame[:, x:x+1] = self.floor_col * 0.85
        for y in range(offset_y, H, tile_size):
            frame[y:y+1, :] = self.floor_col * 0.85

        # Torches (with flicker light)
        for (tx_frac, ty_frac) in self.torches:
            tx = int((tx_frac - self.cam_x * 0.25) * W)
            ty = int((ty_frac - self.cam_y * 0.25) * H)
            tx = max(0, min(W - 1, tx))
            ty = max(0, min(H - 1, ty))
            r = max(3, int(W * 0.015))
            # Glow
            for dy in range(-r*4, r*4):
                for dx in range(-r*4, r*4):
                    d = math.sqrt(dx**2 + dy**2) / (r * 4)
                    if d < 1.0:
                        gy2 = ty + dy
                        gx2 = tx + dx
                        if 0 <= gy2 < H and 0 <= gx2 < W:
                            frame[gy2, gx2] += self.torch_col * (1 - d) * 0.3
            # Torch center
            y0 = max(0, ty - r)
            y1 = min(H, ty + r)
            x0 = max(0, tx - r)
            x1 = min(W, tx + r)
            frame[y0:y1, x0:x1] = self.torch_col

        return _clamp(frame)


SCENE_CLASSES = [CorridorScene, OutdoorScene, PlatformerScene, DungeonScene]


# ── Video generator ────────────────────────────────────────────────

def _generate_action_sequence(num_frames: int, rng) -> list[int]:
    """Generate a realistic action sequence for one clip.

    Actions are held for a random number of frames (like a real player pressing
    and holding a button), with brief idle breaks in between.
    """
    actions = []
    available = [
        ACTION_FORWARD, ACTION_BACKWARD, ACTION_LEFT, ACTION_RIGHT,
        ACTION_RUN, ACTION_JUMP, ACTION_IDLE, ACTION_INTERACT,
    ]
    # Weights: forward/run/left/right most common, jump occasional, interact rare
    weights = [0.25, 0.08, 0.15, 0.15, 0.12, 0.10, 0.10, 0.05]

    while len(actions) < num_frames:
        action = rng.choice(available, p=weights)
        # Hold duration: 2-10 frames, jumps shorter (3-6), idle shorter (1-4)
        if action == ACTION_JUMP:
            hold = int(rng.integers(3, 7))
        elif action == ACTION_IDLE:
            hold = int(rng.integers(1, 5))
        elif action == ACTION_INTERACT:
            hold = int(rng.integers(2, 5))
        else:
            hold = int(rng.integers(3, 12))
        actions.extend([action] * hold)

    return [int(a) for a in actions[:num_frames]]


def _add_noise(frame: np.ndarray, noise_scale: float = 4.0) -> np.ndarray:
    """Add slight pixel noise to reduce synthetic sharpness."""
    noise = np.random.randn(*frame.shape).astype(np.float32) * noise_scale
    return _clamp(frame.astype(np.float32) + noise)


def _add_interact_flash(frame: np.ndarray, t: float = 0.3) -> np.ndarray:
    """Add white flash overlay for INTERACT action."""
    flash = np.ones_like(frame, dtype=np.float32) * 255.0
    blended = frame.astype(np.float32) * (1 - t) + flash * t
    return _clamp(blended)


def generate_clip(
    scene_cls,
    num_frames: int,
    resolution: int,
    seed: int,
    noise_scale: float = 3.0,
) -> tuple[list[np.ndarray], list[int]]:
    """Generate one video clip. Returns (frames, actions).

    frames: list of (H, W, 3) uint8 numpy arrays
    actions: list of int, len == num_frames
    """
    rng = np.random.default_rng(seed)
    scene = scene_cls(resolution, resolution, seed)
    actions = _generate_action_sequence(num_frames, rng)

    frames = []
    for t, action in enumerate(actions):
        scene.apply_action(action, dt=1.0)
        frame = scene.render()
        frame = _add_noise(frame, noise_scale)
        if action == ACTION_INTERACT:
            flash_t = 0.25 + rng.uniform(-0.1, 0.1)
            frame = _add_interact_flash(frame, flash_t)
        frames.append(frame)

    return frames, actions


def save_clip(
    frames: list[np.ndarray],
    actions: list[int],
    out_dir: Path,
    clip_name: str,
    fps: float,
):
    """Save frames as mp4 and actions as JSON."""
    mp4_path = out_dir / f"{clip_name}.mp4"
    json_path = out_dir / f"{clip_name}_actions.json"

    writer = imageio.get_writer(str(mp4_path), fps=fps, codec="libx264",
                                quality=6, pixelformat="yuv420p")
    for frame in frames:
        writer.append_data(frame)
    writer.close()

    with open(json_path, "w") as f:
        json.dump(actions, f)


# ── Main ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate action-conditioned synthetic video for Stage 2 training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--num_videos", type=int, default=200,
                        help="Number of clips to generate")
    parser.add_argument("--output_dir", default="data/videos/stage2_synthetic",
                        help="Output directory (gets mp4 + _actions.json pairs)")
    parser.add_argument("--num_frames", type=int, default=40,
                        help="Frames per clip")
    parser.add_argument("--resolution", type=int, default=256,
                        help="Square resolution (256 for training)")
    parser.add_argument("--fps", type=float, default=10.0,
                        help="Video FPS (10 = good balance for training)")
    parser.add_argument("--noise_scale", type=float, default=3.0,
                        help="Pixel noise to add per frame (reduces synthetic sharpness)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing clips")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test: 20 clips, 16 frames, 128px")
    parser.add_argument("--seed", type=int, default=None,
                        help="Global seed for reproducibility (None = random)")
    args = parser.parse_args()

    if args.quick:
        args.num_videos = 20
        args.num_frames = 16
        args.resolution = 128
        args.fps = 10.0

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_seed = args.seed if args.seed is not None else random.randint(0, 2**31)
    rng = np.random.default_rng(base_seed)

    print(f"\n=== Action-Conditioned Synthetic Video Generator ===")
    print(f"Output:       {out_dir}")
    print(f"Clips:        {args.num_videos}")
    print(f"Frames/clip:  {args.num_frames}")
    print(f"Resolution:   {args.resolution}×{args.resolution}")
    print(f"FPS:          {args.fps}")
    print(f"Base seed:    {base_seed}")
    print()

    from collections import Counter
    action_counts = Counter()
    generated = 0
    skipped = 0

    for i in range(args.num_videos):
        clip_name = f"synth_action_{i:05d}"
        mp4_path = out_dir / f"{clip_name}.mp4"
        json_path = out_dir / f"{clip_name}_actions.json"

        if not args.overwrite and mp4_path.exists() and json_path.exists():
            skipped += 1
            continue

        # Pick a random scene class
        scene_cls = SCENE_CLASSES[i % len(SCENE_CLASSES)]
        clip_seed = int(rng.integers(0, 2**31))

        frames, actions = generate_clip(
            scene_cls=scene_cls,
            num_frames=args.num_frames,
            resolution=args.resolution,
            seed=clip_seed,
            noise_scale=args.noise_scale,
        )

        save_clip(frames, actions, out_dir, clip_name, args.fps)
        action_counts.update(actions)
        generated += 1

        if (i + 1) % 50 == 0 or i == args.num_videos - 1:
            print(f"  [{i+1}/{args.num_videos}] {generated} generated, {skipped} skipped")

    # Print action distribution
    total = sum(action_counts.values())
    print(f"\nAction distribution ({total} total frames):")
    for idx, name in enumerate(["idle", "forward", "backward", "left", "right",
                                  "run", "jump", "interact"]):
        pct = action_counts.get(idx, 0) / max(1, total) * 100
        bar = "█" * int(pct / 2)
        print(f"  {idx} {name:<10} {pct:5.1f}%  {bar}")

    print(f"\n{'='*50}")
    print(f"Generated {generated} clips in {out_dir}")
    if skipped:
        print(f"Skipped {skipped} existing clips (use --overwrite to regenerate)")
    print(f"\nTo train Stage 2:")
    print(f"  python scripts/train_zimage_stage2.py \\")
    print(f"      --data_dir {out_dir} \\")
    print(f"      --stage1_checkpoint checkpoints/zimage_stage1_v3/world_model_final.pt \\")
    print(f"      --epochs 30")


if __name__ == "__main__":
    main()
