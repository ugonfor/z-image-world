#!/usr/bin/env python3
"""
Stage 2 Training: Action Conditioning with LoRA

Fine-tunes the causal DiT to respond to actions using LoRA.
Trains on video-action pairs.

Usage:
    python scripts/train_stage2.py \
        --data_path data/action_videos \
        --checkpoint checkpoints/stage1/final.pt \
        --output_dir checkpoints/stage2
"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import yaml

# Add parent directory to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import CausalDiT, ActionEncoder, StreamVAE
from training import ActionFinetuner, ActionFinetuneConfig
from data import ActionVideoDataset, ActionVideoCollator


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 2: Action Fine-tuning")

    # Data
    parser.add_argument("--data_path", type=str, required=True, help="Path to action-video data")
    parser.add_argument("--num_frames", type=int, default=8, help="Frames per sample")
    parser.add_argument("--resolution", type=int, nargs=2, default=[480, 640], help="Resolution (H W)")

    # Training
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--max_steps", type=int, default=50000, help="Max training steps")

    # LoRA
    parser.add_argument("--lora_rank", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")

    # Model
    parser.add_argument("--checkpoint", type=str, required=True, help="Stage 1 checkpoint")

    # Output
    parser.add_argument("--output_dir", type=str, default="checkpoints/stage2", help="Output directory")
    parser.add_argument("--save_steps", type=int, default=2000, help="Save every N steps")

    # Logging
    parser.add_argument("--wandb_project", type=str, default="z-image-world", help="W&B project")
    parser.add_argument("--wandb_run", type=str, default="stage2_action", help="W&B run name")

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

    print("Loading Stage 1 model...")

    # Load Stage 1 model
    checkpoint = torch.load(args.checkpoint, map_location=device)

    model = CausalDiT(
        in_channels=16,
        hidden_dim=4096,
        num_heads=32,
        num_layers=28,
        num_frames=args.num_frames,
        action_injection_layers=[7, 14, 21],
    )

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    print("Initializing action encoder...")

    # Initialize action encoder
    action_encoder = ActionEncoder(
        num_actions=17,
        embedding_dim=512,
        hidden_dim=4096,
        num_frames=args.num_frames,
    )

    # Initialize VAE (placeholder)
    print("Loading VAE...")
    vae = StreamVAE(tile_size=512)

    print("Creating dataset...")

    # Create dataset
    dataset = ActionVideoDataset(
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
        collate_fn=ActionVideoCollator(),
    )

    print(f"Dataset size: {len(dataset)} samples")

    # Create fine-tuner config
    config = ActionFinetuneConfig(
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        use_curriculum=True,
    )

    # Create fine-tuner
    finetuner = ActionFinetuner(
        model=model,
        action_encoder=action_encoder,
        vae=vae,
        config=config,
        device=device,
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
            print(f"Step {step}: loss={metrics['loss']:.4f}, action_loss={metrics.get('action_loss', 0):.4f}")

    print("Starting training...")

    # Training loop
    epoch = 0
    while finetuner.global_step < args.max_steps:
        metrics = finetuner.train_epoch(dataloader, log_fn=log_fn)
        epoch += 1

        print(f"Epoch {epoch} complete: avg_loss={metrics['loss']:.4f}")

        # Save checkpoint
        if finetuner.global_step % args.save_steps == 0 or finetuner.global_step >= args.max_steps:
            checkpoint_path = output_dir / f"checkpoint_{finetuner.global_step}.pt"
            finetuner.save_checkpoint(str(checkpoint_path))
            print(f"Saved checkpoint to {checkpoint_path}")

    # Save final model
    finetuner.save_checkpoint(str(output_dir / "final.pt"))

    # Export LoRA weights separately
    finetuner.export_lora(str(output_dir / "lora_weights.pt"))

    print("Training complete!")


if __name__ == "__main__":
    main()
