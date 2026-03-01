#!/usr/bin/env python3
"""
Run FIFO-Diffusion video generation with ZImageWorldModel.

Generates a video from a text prompt using the FIFO-Diffusion pipeline.
Compares checkpoints side-by-side when --compare is used.

Usage:
    # Generate 24-frame video with latest v2 checkpoint
    uv run python scripts/run_fifo.py --prompt "a forest path at golden hour"

    # Use v3 (richer training data) checkpoint
    uv run python scripts/run_fifo.py \
        --checkpoint checkpoints/zimage_stage1_v3/world_model_final.pt \
        --prompt "a city street in heavy rain"

    # High quality with CFG (2x slower, more text-aligned)
    uv run python scripts/run_fifo.py --use_cfg \
        --prompt "a beach at sunset, waves crashing"

    # Compare v2 vs v3
    uv run python scripts/run_fifo.py --compare \
        --checkpoint checkpoints/zimage_stage1_v3/world_model_final.pt \
        --prompt "a mountain lake in autumn"
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def inspect_checkpoint(ckpt_path: str) -> None:
    """Print gamma values and loss info from a checkpoint."""
    import torch
    ckpt = torch.load(ckpt_path, map_location="cpu")
    g = ckpt["temporal_state_dict"]
    gammas = {k: v.item() for k, v in g.items() if "gamma" in k and v.numel() == 1}
    if gammas:
        vals = list(gammas.values())
        print(f"  Gammas: min={min(vals):.4f}, max={max(vals):.4f}, mean={sum(vals)/len(vals):.4f}")
        # Show first 4
        for k, v in list(gammas.items())[:4]:
            print(f"    {k}: {v:.4f}")
    print(f"  Epoch: {ckpt.get('epoch', '?')}")


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
    guidance_scale: float,
    seed: int,
    output_path: str,
    save_mp4: bool,
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
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = FIFOPipeline.from_pretrained(
        model_path=model_path,
        checkpoint=checkpoint,
        config=cfg,
        device=device,
    )

    frames = pipeline.generate(prompt=prompt, num_frames=num_frames, seed=seed)

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
                        help="Temporal layer checkpoint path. Defaults to latest available.")
    parser.add_argument("--prompt", default="a lush green forest path in afternoon sunlight")
    parser.add_argument("--num_frames", type=int, default=24)
    parser.add_argument("--queue_size", type=int, default=8)
    parser.add_argument("--num_inference_steps", type=int, default=32)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--anchor_noise_frac", type=float, default=0.75,
                        help="0.5=more coherent/less motion, 0.9=less coherent/more motion")
    parser.add_argument("--use_cfg", action="store_true",
                        help="Enable Classifier-Free Guidance (2x slower, better text adherence)")
    parser.add_argument("--guidance_scale", type=float, default=3.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="output_fifo.gif")
    parser.add_argument("--save_mp4", action="store_true", help="Also save as MP4")
    parser.add_argument("--compare", action="store_true",
                        help="Compare checkpoint vs no-checkpoint (base model)")
    parser.add_argument("--inspect", action="store_true",
                        help="Print checkpoint gamma values and exit")
    args = parser.parse_args()

    # Default checkpoint: find latest v3, then v2
    if args.checkpoint is None:
        candidates = [
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

    if args.compare:
        # Generate with and without checkpoint for direct comparison
        out_base = Path(args.output)
        out_ckpt = out_base.with_stem(out_base.stem + "_ckpt")
        out_none = out_base.with_stem(out_base.stem + "_none")

        print(f"\n=== Run 1: With checkpoint ({args.checkpoint}) ===")
        if args.checkpoint:
            inspect_checkpoint(args.checkpoint)
        run_pipeline(
            model_path=args.model_path,
            checkpoint=args.checkpoint,
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
            output_path=str(out_ckpt),
            save_mp4=args.save_mp4,
        )

        print(f"\n=== Run 2: No checkpoint (untrained temporal layers) ===")
        run_pipeline(
            model_path=args.model_path,
            checkpoint=None,
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
            output_path=str(out_none),
            save_mp4=args.save_mp4,
        )

        print(f"\nComparison saved:")
        print(f"  Checkpoint: {out_ckpt}")
        print(f"  No checkpoint: {out_none}")
    else:
        if args.checkpoint:
            print(f"\nCheckpoint info:")
            inspect_checkpoint(args.checkpoint)
        run_pipeline(
            model_path=args.model_path,
            checkpoint=args.checkpoint,
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
            output_path=args.output,
            save_mp4=args.save_mp4,
        )


if __name__ == "__main__":
    main()
