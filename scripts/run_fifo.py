#!/usr/bin/env python3
"""
Run FIFO-Diffusion video generation with ZImageWorldModel.

Generates a video from a text prompt using the FIFO-Diffusion pipeline.
Supports action conditioning (Stage 2 checkpoint required for --use_actions).

Action indices:
  0=idle  1=forward  2=backward  3=left  4=right  5=run  6=jump  7=interact

Usage:
    # Basic generation (auto-picks latest checkpoint)
    python scripts/run_fifo.py --prompt "a forest path at golden hour"

    # Stage 2 with action conditioning
    python scripts/run_fifo.py \\
        --checkpoint checkpoints/zimage_stage2_gamefactory/world_model_s2_final.pt \\
        --use_actions --action_pattern forward \\
        --prompt "a Minecraft world with green hills"

    # Custom action sequence (repeats to fill num_frames)
    python scripts/run_fifo.py --use_actions --actions 1,1,1,1,6,0,1,1 \\
        --prompt "a forest path"

    # Compare: action_forward vs action_backward vs no_action
    python scripts/run_fifo.py --action_compare \\
        --prompt "a Minecraft landscape"

    # CFG (2x slower, better text alignment)
    python scripts/run_fifo.py --use_cfg --prompt "a beach at sunset"

    # Inspect checkpoint gammas
    python scripts/run_fifo.py --inspect \\
        --checkpoint checkpoints/zimage_stage2_gamefactory/world_model_s2_final.pt
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

ACTION_NAMES = {
    0: "idle", 1: "forward", 2: "backward", 3: "left",
    4: "right", 5: "run", 6: "jump", 7: "interact",
}
ACTION_PATTERNS = {
    "forward":  [1] * 24,
    "backward": [2] * 24,
    "left":     [3] * 24,
    "right":    [4] * 24,
    "run":      [5] * 24,
    "jump":     [6, 6, 0, 0, 6, 6, 0, 0] * 3,
    "idle":     [0] * 24,
    "mixed":    [1,1,1,1,6,0,1,1,3,3,1,1,1,1,6,0,5,5,5,5,1,1,2,2],
}


def _parse_actions(actions_str: str, pattern: str, num_frames: int) -> list[int]:
    """Parse action sequence from string or named pattern, padded to num_frames."""
    if actions_str:
        seq = [int(x.strip()) for x in actions_str.split(",")]
    elif pattern and pattern in ACTION_PATTERNS:
        seq = ACTION_PATTERNS[pattern]
    else:
        return None

    # Tile to fill num_frames
    if len(seq) < num_frames:
        reps = (num_frames + len(seq) - 1) // len(seq)
        seq = (seq * reps)[:num_frames]
    return seq[:num_frames]


def inspect_checkpoint(ckpt_path: str) -> dict:
    """Print gamma values and action layer info from a checkpoint."""
    import torch
    ckpt = torch.load(ckpt_path, map_location="cpu")
    g = ckpt["temporal_state_dict"]
    gammas = {k: v.item() for k, v in g.items() if "gamma" in k and v.numel() == 1}
    result = {}
    if gammas:
        vals = list(gammas.values())
        abs_vals = [abs(v) for v in vals]
        result["gamma_abs_mean"] = sum(abs_vals) / len(abs_vals)
        result["gamma_max"] = max(abs_vals)
        print(f"  Gammas: abs-mean={result['gamma_abs_mean']:.4f}, max={result['gamma_max']:.4f}")
        for k, v in list(gammas.items())[:4]:
            print(f"    {k}: {v:.4f}")
    has_actions = "action_injections_state_dict" in ckpt
    result["has_actions"] = has_actions
    print(f"  Epoch: {ckpt.get('epoch', '?')}")
    print(f"  Action layers: {'YES (Stage 2)' if has_actions else 'NO (Stage 1)'}")
    return result


def run_pipeline(
    model_path: str,
    checkpoint: str | None,
    prompt: str,
    num_frames: int,
    queue_size: int,
    num_inference_steps: int,
    height: int,
    width: int,
    anchor_noise_frac: float,
    use_cfg: bool,
    use_actions: bool,
    guidance_scale: float,
    seed: int,
    output_path: str,
    save_mp4: bool,
    actions: list[int] | None = None,
) -> list:
    import torch
    from inference.fifo_pipeline import FIFOConfig, FIFOPipeline

    cfg = FIFOConfig(
        queue_size=queue_size,
        num_inference_steps=num_inference_steps,
        height=height,
        width=width,
        guidance_scale=guidance_scale,
        anchor_init=True,
        anchor_noise_frac=anchor_noise_frac,
        use_cfg=use_cfg,
        use_actions=use_actions,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = FIFOPipeline.from_pretrained(
        model_path=model_path,
        checkpoint=checkpoint,
        config=cfg,
        device=device,
    )

    frames = pipeline.generate(
        prompt=prompt, num_frames=num_frames, seed=seed, actions=actions,
    )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    FIFOPipeline.save_gif(frames, out, fps=6.0)
    if save_mp4:
        FIFOPipeline.save_video(frames, out.with_suffix(".mp4"), fps=6.0)

    return frames


def main():
    parser = argparse.ArgumentParser(description="FIFO video generation")
    parser.add_argument("--model_path", default="weights/Z-Image-Turbo")
    parser.add_argument("--checkpoint", default=None,
                        help="Checkpoint path. Defaults to latest Stage 2 then Stage 1.")
    parser.add_argument("--prompt", default="a lush green Minecraft landscape at sunrise")
    parser.add_argument("--num_frames", type=int, default=24)
    parser.add_argument("--queue_size", type=int, default=8)
    parser.add_argument("--num_inference_steps", type=int, default=32)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--anchor_noise_frac", type=float, default=0.75)
    parser.add_argument("--use_cfg", action="store_true",
                        help="Classifier-Free Guidance (2x slower, better text adherence)")
    parser.add_argument("--use_actions", action="store_true",
                        help="Enable action conditioning (requires Stage 2 checkpoint)")
    parser.add_argument("--actions", default=None,
                        help="Comma-separated action indices, e.g. '1,1,6,0,1,1' (tiled to num_frames)")
    parser.add_argument("--action_pattern", default=None,
                        choices=list(ACTION_PATTERNS.keys()),
                        help="Named action pattern (used if --actions not set)")
    parser.add_argument("--action_compare", action="store_true",
                        help="Generate forward/backward/no-action variants for comparison")
    parser.add_argument("--guidance_scale", type=float, default=3.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="output_fifo.gif")
    parser.add_argument("--save_mp4", action="store_true")
    parser.add_argument("--compare", action="store_true",
                        help="Compare Stage 2 checkpoint vs Stage 1 (no action)")
    parser.add_argument("--inspect", action="store_true",
                        help="Print checkpoint info and exit")
    args = parser.parse_args()

    # Default checkpoint: Stage 2 > Stage 1 v3 > Stage 1 v2
    if args.checkpoint is None:
        candidates = [
            "checkpoints/zimage_stage2_gamefactory/world_model_s2_final.pt",
            "checkpoints/zimage_stage1_v3/world_model_final.pt",
            "checkpoints/zimage_stage1_v2/world_model_final.pt",
        ]
        for c in candidates:
            if Path(c).exists():
                args.checkpoint = c
                print(f"Auto-selected checkpoint: {c}")
                break

    if args.inspect and args.checkpoint:
        print(f"\nInspecting {args.checkpoint}:")
        inspect_checkpoint(args.checkpoint)
        return

    # Parse action sequence
    action_seq = _parse_actions(args.actions, args.action_pattern, args.num_frames)

    common = dict(
        model_path=args.model_path,
        prompt=args.prompt,
        num_frames=args.num_frames,
        queue_size=args.queue_size,
        num_inference_steps=args.num_inference_steps,
        height=args.height,
        width=args.width,
        anchor_noise_frac=args.anchor_noise_frac,
        use_cfg=args.use_cfg,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        save_mp4=args.save_mp4,
    )

    if args.action_compare:
        # Generate forward / backward / no-action for side-by-side comparison
        out_base = Path(args.output)
        ckpt = args.checkpoint
        if args.checkpoint:
            print(f"\nCheckpoint:")
            info = inspect_checkpoint(args.checkpoint)

        for pattern_name, act_list in [
            ("forward",   ACTION_PATTERNS["forward"]),
            ("backward",  ACTION_PATTERNS["backward"]),
            ("jump",      ACTION_PATTERNS["jump"]),
            ("no_action", None),
        ]:
            out_path = out_base.with_stem(out_base.stem + f"_{pattern_name}")
            use_act = (act_list is not None)
            print(f"\n=== {pattern_name} (use_actions={use_act}) ===")
            run_pipeline(
                checkpoint=ckpt,
                use_actions=use_act,
                actions=act_list[:args.num_frames] if act_list else None,
                output_path=str(out_path),
                **common,
            )
            print(f"  Saved: {out_path}")

        print(f"\nAll comparison GIFs saved with prefix: {out_base.stem}_*")

    elif args.compare:
        # Stage 2 vs Stage 1 comparison
        out_base = Path(args.output)
        out_s2 = out_base.with_stem(out_base.stem + "_stage2")
        out_s1 = out_base.with_stem(out_base.stem + "_stage1")

        s2_ckpt = "checkpoints/zimage_stage2_gamefactory/world_model_s2_final.pt"
        s1_ckpt = "checkpoints/zimage_stage1_v3/world_model_final.pt"

        print(f"\n=== Stage 2 (action conditioned) ===")
        run_pipeline(
            checkpoint=s2_ckpt if Path(s2_ckpt).exists() else args.checkpoint,
            use_actions=True,
            actions=ACTION_PATTERNS["forward"][:args.num_frames],
            output_path=str(out_s2),
            **common,
        )

        print(f"\n=== Stage 1 (no action) ===")
        run_pipeline(
            checkpoint=s1_ckpt if Path(s1_ckpt).exists() else None,
            use_actions=False,
            actions=None,
            output_path=str(out_s1),
            **common,
        )

        print(f"\nComparison: {out_s2}  vs  {out_s1}")

    else:
        if args.checkpoint:
            print(f"\nCheckpoint info:")
            inspect_checkpoint(args.checkpoint)
        run_pipeline(
            checkpoint=args.checkpoint,
            use_actions=args.use_actions,
            actions=action_seq,
            output_path=args.output,
            **common,
        )


if __name__ == "__main__":
    main()
