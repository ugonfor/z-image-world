#!/usr/bin/env python3
"""
Measure action responsiveness of FIFO-generated videos.

Compares forward vs backward videos to quantify how much the action
conditioning changes the output relative to unrelated noise variation.

Metrics:
  action_diff:   mean pixel-L1 between forward and backward video
  baseline_diff: mean pixel-L1 between forward and no-action video
  responsiveness_ratio: action_diff / (action_diff + baseline_diff + eps)
    - 0.5 = equal difference (no conditioning signal)
    - > 0.5 = action has more effect than noise
    - < 0.5 = noise dominates

Usage:
    python scripts/measure_action_responsiveness.py \
        --forward  outputs/action_compare/action_compare_forward.gif \
        --backward outputs/action_compare/action_compare_backward.gif \
        --no_action outputs/action_compare/action_compare_no_action.gif \
        [--jump outputs/action_compare/action_compare_jump.gif]
"""

import argparse
import sys
from pathlib import Path

import numpy as np


def load_gif_frames(path: str) -> np.ndarray:
    """Load GIF as float32 array (N, H, W, 3) in [0, 1]."""
    from PIL import Image
    gif = Image.open(path)
    frames = []
    try:
        while True:
            frame = gif.copy().convert("RGB")
            frames.append(np.array(frame, dtype=np.float32) / 255.0)
            gif.seek(gif.tell() + 1)
    except EOFError:
        pass
    return np.stack(frames, axis=0)


def mean_l1(a: np.ndarray, b: np.ndarray) -> float:
    """Mean absolute pixel difference between two video arrays, truncated to shorter."""
    n = min(len(a), len(b))
    return float(np.mean(np.abs(a[:n] - b[:n])))


def frame_diffs(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-frame L1 difference."""
    n = min(len(a), len(b))
    return np.mean(np.abs(a[:n] - b[:n]), axis=(1, 2, 3))


def report(label: str, a: np.ndarray, b: np.ndarray) -> float:
    diffs = frame_diffs(a, b)
    mean_d = float(np.mean(diffs))
    print(f"  {label}: mean_L1={mean_d:.4f}  "
          f"[min={diffs.min():.4f} max={diffs.max():.4f}]")
    return mean_d


def main():
    parser = argparse.ArgumentParser(description="Measure action responsiveness")
    parser.add_argument("--forward",   required=True, help="Forward action GIF")
    parser.add_argument("--backward",  required=True, help="Backward action GIF")
    parser.add_argument("--no_action", required=True, help="No-action GIF")
    parser.add_argument("--jump",      default=None,  help="Jump action GIF (optional)")
    args = parser.parse_args()

    for p in [args.forward, args.backward, args.no_action]:
        if not Path(p).exists():
            print(f"ERROR: file not found: {p}", file=sys.stderr)
            sys.exit(1)

    print("\nLoading GIFs...")
    fwd  = load_gif_frames(args.forward)
    bwd  = load_gif_frames(args.backward)
    none = load_gif_frames(args.no_action)
    print(f"  forward:   {fwd.shape}  (N, H, W, C)")
    print(f"  backward:  {bwd.shape}")
    print(f"  no_action: {none.shape}")

    if args.jump and Path(args.jump).exists():
        jmp = load_gif_frames(args.jump)
        print(f"  jump:      {jmp.shape}")
    else:
        jmp = None

    print("\n=== Pairwise pixel-L1 differences ===")
    fwd_vs_bwd  = report("forward  vs backward ", fwd, bwd)
    fwd_vs_none = report("forward  vs no_action", fwd, none)
    bwd_vs_none = report("backward vs no_action", bwd, none)
    if jmp is not None:
        jmp_vs_fwd  = report("jump     vs forward  ", jmp, fwd)
        jmp_vs_none = report("jump     vs no_action", jmp, none)

    # Responsiveness ratio: action_diff vs noise_diff
    # action_diff = fwd_vs_bwd (two distinct actions)
    # noise_diff  = fwd_vs_none (one action vs baseline)
    eps = 1e-6
    responsiveness = fwd_vs_bwd / (fwd_vs_bwd + fwd_vs_none + eps)

    print(f"\n=== Summary ===")
    print(f"  action_diff   (fwd vs bwd):  {fwd_vs_bwd:.4f}")
    print(f"  baseline_diff (fwd vs none): {fwd_vs_none:.4f}")
    print(f"  responsiveness_ratio:        {responsiveness:.4f}  "
          f"(>0.5 = action > noise)")

    if responsiveness > 0.55:
        verdict = "PASS - action conditioning is clearly working"
    elif responsiveness > 0.50:
        verdict = "MARGINAL - slight action conditioning signal"
    else:
        verdict = "FAIL - action has less effect than noise (conditioning not learned)"
    print(f"  verdict: {verdict}")

    # Also check self-consistency: temporal coherence of each video
    print(f"\n=== Temporal coherence (consecutive frame diff, lower = smoother) ===")
    for name, vid in [("forward", fwd), ("backward", bwd), ("no_action", none)]:
        if len(vid) > 1:
            consec = float(np.mean(np.abs(np.diff(vid, axis=0))))
            print(f"  {name:<12}: consec_L1={consec:.4f}")

    print()
    return responsiveness


if __name__ == "__main__":
    main()
