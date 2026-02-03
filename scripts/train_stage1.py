#!/usr/bin/env python3
"""
Stage 1 Training: Causal Adaptation

Converts Z-Image to autoregressive generation using Diffusion Forcing.
Trains on video clips without action labels.

Usage:
    python scripts/train_stage1.py --data_path data/videos --output_dir checkpoints/stage1
"""

import argparse
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import yaml

# Add parent directory to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import CausalDiT, StreamVAE
from training import DiffusionForcingTrainer, DiffusionForcingConfig
from data import VideoOnlyDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 1: Causal Adaptation Training")

    # Data
    parser.add_argument("--data_path", type=str, required=True, help="Path to video data")
    parser.add_argument("--num_frames", type=int, default=8, help="Frames per sample")
    parser.add_argument("--resolution", type=int, nargs=2, default=[480, 640], help="Resolution (H W)")

    # Training
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--gradient_accumulation", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--max_steps", type=int, default=100000, help="Max training steps")
    parser.add_argument("--warmup_steps", type=int, default=1000, help="Warmup steps")

    # Model
    parser.add_argument("--pretrained", type=str, default=None, help="Path to pretrained model")

    # Output
    parser.add_argument("--output_dir", type=str, default="checkpoints/stage1", help="Output directory")
    parser.add_argument("--save_steps", type=int, default=5000, help="Save every N steps")

    # Logging
    parser.add_argument("--wandb_project", type=str, default="z-image-world", help="W&B project")
    parser.add_argument("--wandb_run", type=str, default="stage1_causal", help="W&B run name")

    # Hardware
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--num_workers", type=int, default=4, help="Data loader workers")

    return parser.parse_args()


def main():
    args = parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(vars(args), f)

    device = torch.device(args.device)

    print("Initializing model...")

    # Initialize model
    if args.pretrained:
        model = CausalDiT.from_pretrained(args.pretrained)
    else:
        model = CausalDiT(
            in_channels=16,
            hidden_dim=4096,
            num_heads=32,
            num_layers=28,
            num_frames=args.num_frames,
            action_injection_layers=[7, 14, 21],
        )

    model = model.to(device)

    # Initialize VAE (placeholder - would load from diffusers)
    print("Loading VAE...")
    vae = StreamVAE(tile_size=512)
    # In practice: vae.set_vae(AutoencoderKL.from_pretrained(...))

    print("Creating dataset...")

    # Create dataset
    dataset = VideoOnlyDataset(
        data_root=args.data_path,
        num_frames=args.num_frames,
        resolution=tuple(args.resolution),
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    print(f"Dataset size: {len(dataset)} samples")

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.01,
        betas=(0.9, 0.999),
    )

    # Create scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.max_steps,
        eta_min=args.learning_rate * 0.1,
    )

    # Create trainer config
    config = DiffusionForcingConfig(
        num_frames=args.num_frames,
        independent_noise=True,
        noise_level_sampling="uniform",
        prediction_type="v_prediction",
    )

    # Create trainer
    trainer = DiffusionForcingTrainer(
        model=model,
        vae=vae,
        config=config,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        gradient_accumulation_steps=args.gradient_accumulation,
    )

    # Initialize wandb
    try:
        import wandb
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run,
            config=vars(args),
        )
        use_wandb = True
    except ImportError:
        print("wandb not available, logging to stdout only")
        use_wandb = False

    def log_fn(metrics, step):
        if use_wandb:
            wandb.log(metrics, step=step)
        if step % 100 == 0:
            print(f"Step {step}: loss={metrics['loss']:.4f}")

    print("Starting training...")

    # Training loop
    epoch = 0
    while trainer.global_step < args.max_steps:
        metrics = trainer.train_epoch(dataloader, log_fn=log_fn)
        epoch += 1

        print(f"Epoch {epoch} complete: avg_loss={metrics['loss']:.4f}")

        # Save checkpoint
        if trainer.global_step % args.save_steps == 0 or trainer.global_step >= args.max_steps:
            checkpoint_path = output_dir / f"checkpoint_{trainer.global_step}.pt"
            trainer.save_checkpoint(str(checkpoint_path))
            print(f"Saved checkpoint to {checkpoint_path}")

    # Save final model
    trainer.save_checkpoint(str(output_dir / "final.pt"))
    print("Training complete!")


if __name__ == "__main__":
    main()
