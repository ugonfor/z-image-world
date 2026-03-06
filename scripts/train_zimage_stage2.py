#!/usr/bin/env python3
"""
Stage 2 Training: Action Conditioning for Z-Image World Model

Loads a Stage 1 checkpoint (temporal layers pre-trained) and fine-tunes
the ActionEncoder + ActionInjection layers with action-labeled video.

Keeps temporal layers frozen initially (to preserve learned dynamics),
then optionally unfreezes them for joint fine-tuning.

Usage:
    # Basic Stage 2 training
    python scripts/train_zimage_stage2.py \
        --model_path weights/Z-Image-Turbo \
        --stage1_checkpoint checkpoints/zimage_stage1_v2/world_model_final.pt \
        --data_dir data/videos/gameplay \
        --epochs 30

    # With temporal layers unfrozen (joint fine-tuning)
    python scripts/train_zimage_stage2.py \
        --model_path weights/Z-Image-Turbo \
        --stage1_checkpoint checkpoints/zimage_stage1_v2/world_model_final.pt \
        --data_dir data/videos/gameplay \
        --unfreeze_temporal \
        --epochs 20

Data format (two options):
    Option A - Video + JSON:
        gameplay_001.mp4
        gameplay_001_actions.json  # List[int], one action index per frame
        # Action indices: 0=idle, 1=forward, 2=backward, 3=left, 4=right,
        #                 5=run, 6=jump, 7=attack, 8=interact, ...

    Option B - Directory per clip:
        clip_001/
            frames/0000.png, 0001.png, ...
            actions.json  # List[int]
"""

import argparse
import os
import sys
import time
from pathlib import Path

sys.stdout = os.fdopen(sys.stdout.fileno(), "w", buffering=1)
sys.stderr = os.fdopen(sys.stderr.fileno(), "w", buffering=1)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from einops import rearrange

sys.path.insert(0, str(Path(__file__).parent.parent))

from training.diffusion_forcing import DiffusionForcingConfig, DiffusionForcingLoss
from training.action_finetune import ActionConditioningLoss


class ActionVideoDataset(Dataset):
    """Dataset loading video-action pairs for Stage 2 training.

    Supports two formats:
    1. video.mp4 + video_actions.json (NitroGen style)
    2. clip_dir/frames/*.png + clip_dir/actions.json (directory style)
    """

    def __init__(self, data_dir: str, num_frames: int = 4, resolution: int = 256):
        import json
        self.data_dir = Path(data_dir)
        self.num_frames = num_frames
        self.resolution = resolution
        self.samples = []

        # Find video+action pairs
        for p in sorted(self.data_dir.rglob("*.mp4")):
            action_file = p.with_name(f"{p.stem}_actions.json")
            if action_file.exists():
                with open(action_file) as f:
                    actions = json.load(f)
                self.samples.append({"type": "video", "video": p, "actions": actions})

        # Find directory-style clips
        for d in sorted(self.data_dir.iterdir()):
            if d.is_dir():
                frames_dir = d / "frames"
                action_file = d / "actions.json"
                if frames_dir.exists() and action_file.exists():
                    with open(action_file) as f:
                        actions = json.load(f)
                    frame_files = sorted(
                        list(frames_dir.glob("*.png")) + list(frames_dir.glob("*.jpg"))
                    )
                    if frame_files:
                        self.samples.append({
                            "type": "directory",
                            "frames": frame_files,
                            "actions": actions,
                        })

        if not self.samples:
            raise ValueError(
                f"No action-labeled video samples found in {data_dir}\n"
                f"Expected: video.mp4 + video_actions.json, OR\n"
                f"          clip_dir/frames/*.png + clip_dir/actions.json"
            )

        print(f"Found {len(self.samples)} action-labeled clips")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        import imageio
        import numpy as np
        from PIL import Image as PILImage

        sample = self.samples[idx]

        if sample["type"] == "video":
            reader = imageio.get_reader(str(sample["video"]), "ffmpeg")
            total = reader.count_frames()
            all_actions = sample["actions"]
        else:
            total = len(sample["frames"])
            all_actions = sample["actions"]

        # Random start
        max_start = max(0, total - self.num_frames)
        start = torch.randint(0, max_start + 1, (1,)).item() if max_start > 0 else 0

        frames, actions = [], []
        for fi in range(start, start + self.num_frames):
            fi_clamp = min(fi, total - 1)
            if sample["type"] == "video":
                try:
                    frame = reader.get_data(fi_clamp)
                except Exception:
                    frame = reader.get_data(0)
            else:
                import numpy as _np
                frame = _np.array(PILImage.open(str(sample["frames"][fi_clamp])))

            pil = PILImage.fromarray(frame).resize(
                (self.resolution, self.resolution), PILImage.BILINEAR
            )
            frames.append(torch.from_numpy(np.array(pil)).float() / 255.0)
            action_idx = all_actions[fi_clamp] if fi_clamp < len(all_actions) else 0
            actions.append(action_idx)

        if sample["type"] == "video":
            reader.close()

        frames = rearrange(torch.stack(frames), "f h w c -> f c h w")
        actions = torch.tensor(actions, dtype=torch.long)
        return {"frames": frames, "actions": actions}


def train_stage2(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info(0)
        print(f"GPU Memory: {free / 1e9:.1f}GB free / {total / 1e9:.1f}GB total")

    # --- Load model ---
    print("\n=== Loading Z-Image World Model ===")
    from models.zimage_world_model import ZImageWorldModel

    model = ZImageWorldModel.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        temporal_every_n=args.temporal_every_n,
        freeze_spatial=True,
        device=device,
    )

    # Load Stage 1 checkpoint
    if args.stage1_checkpoint and Path(args.stage1_checkpoint).exists():
        print(f"Loading Stage 1 checkpoint: {args.stage1_checkpoint}")
        ckpt = torch.load(args.stage1_checkpoint, map_location=device)
        model.temporal_layers.load_state_dict(ckpt["temporal_state_dict"])
        print(f"  Stage 1 temporal layers loaded (epoch {ckpt.get('epoch', '?')})")
    else:
        print("WARNING: No Stage 1 checkpoint provided. Training from scratch.")

    # Freeze temporal layers initially (optional)
    if not args.unfreeze_temporal:
        print("Temporal layers frozen (Stage 2 trains action layers only)")
        for p in model.temporal_layers.parameters():
            p.requires_grad_(False)
    else:
        print("Temporal layers unfrozen (joint training)")
        for p in model.temporal_layers.parameters():
            p.requires_grad_(True)
        # Cast scalar gamma params to float32 to avoid bfloat16 precision floor.
        # In bfloat16, gamma≈0.025 has step size ~1.2e-4; lr=5e-6 updates ~5e-6 → never updates.
        # float32 precision at 0.025 is ~6e-9, always below any gradient update.
        n_cast = 0
        for name, p in model.temporal_layers.named_parameters():
            if "gamma" in name and p.dtype == torch.bfloat16:
                p.data = p.data.float()
                n_cast += 1
        if n_cast:
            print(f"  Cast {n_cast} gamma params to float32 (bfloat16 precision fix)")

    # Unfreeze action layers
    for p in model.action_injections.parameters():
        p.requires_grad_(True)
    for p in model.action_encoder.parameters():
        p.requires_grad_(True)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {total_params / 1e6:.1f}M")

    model.enable_gradient_checkpointing()

    # --- Dataset ---
    print("\n=== Loading Dataset ===")
    dataset = ActionVideoDataset(
        args.data_dir,
        num_frames=args.num_frames,
        resolution=args.resolution,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    print(f"Dataset: {len(dataset)} clips, batch_size={args.batch_size}")

    # --- Optimizer (parameter groups for independent LRs) ---
    action_params = (
        list(model.action_injections.parameters())
        + list(model.action_encoder.parameters())
    )
    temporal_params = list(model.temporal_layers.parameters())
    lr_temporal = args.lr_temporal
    if args.unfreeze_temporal:
        param_groups = [
            {"params": action_params, "lr": args.lr},
            {"params": temporal_params, "lr": lr_temporal},
        ]
        print(f"Optimizer: action lr={args.lr:.1e}, temporal lr={lr_temporal:.1e}")
    else:
        param_groups = [{"params": action_params, "lr": args.lr}]
        print(f"Optimizer: action lr={args.lr:.1e}")
    optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01, betas=(0.9, 0.999))
    total_steps = args.epochs * len(dataset) // (args.batch_size * args.grad_accum)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(total_steps, 1), eta_min=args.lr * 0.01,
    )

    # --- Loss ---
    df_config = DiffusionForcingConfig(
        num_train_timesteps=1000,
        beta_schedule="scaled_linear",
        prediction_type="v_prediction",
        independent_noise=True,
        noise_level_sampling="pyramid",
        num_frames=args.num_frames,
    )
    loss_fn = DiffusionForcingLoss(df_config).to(device)
    contrastive_weight = args.contrastive_weight
    action_loss_fn = ActionConditioningLoss() if contrastive_weight > 0 else None
    if action_loss_fn:
        print(f"Contrastive action loss enabled (weight={contrastive_weight})")

    # Collect all trainable parameters for gradient clipping
    all_trainable = [p for p in model.parameters() if p.requires_grad]

    # --- Checkpoint directory ---
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Resume
    start_epoch, global_step = 0, 0
    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device)
        model.temporal_layers.load_state_dict(ckpt.get("temporal_state_dict", {}), strict=False)
        model.action_injections.load_state_dict(ckpt.get("action_injections_state_dict", {}), strict=False)
        model.action_encoder.load_state_dict(ckpt.get("action_encoder_state_dict", {}), strict=False)
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0)
        global_step = ckpt.get("global_step", 0)
        print(f"Resumed from epoch {start_epoch}")

    # --- Training ---
    print(f"\n=== Stage 2 Training ({args.epochs} epochs) ===")
    model.train()
    model.transformer.eval()
    model.vae.eval()

    for epoch in range(start_epoch, args.epochs):
        epoch_loss = 0.0
        epoch_steps = 0
        t_start = time.time()

        for batch_idx, batch in enumerate(dataloader):
            frames = batch["frames"].to(device, dtype=torch.bfloat16)
            actions = batch["actions"].to(device)  # (B, F) long
            batch_size = frames.shape[0]

            with torch.no_grad():
                latents = model.encode_frames(frames)  # (B, F, 16, H/8, W/8)

            timesteps = loss_fn.sample_timesteps(batch_size, args.num_frames, device)
            noise = torch.randn_like(latents)
            noisy_latents = loss_fn.add_noise(latents, noise, timesteps)

            with torch.amp.autocast(device, dtype=torch.bfloat16, enabled=(device == "cuda")):
                model_output = model(noisy_latents, timesteps.float(), actions=actions)
                if model_output.dim() == 4 and latents.dim() == 5:
                    model_output = model_output.unsqueeze(1)
                loss_dict = loss_fn(model_output, latents, noise, timesteps)

                # Contrastive action loss: forces different actions → different embeddings
                if action_loss_fn is not None and batch_size >= 2:
                    action_emb = model.action_encoder(actions)  # (B, F, D)
                    c_loss = action_loss_fn._action_consistency_loss(
                        model_output, actions, action_emb
                    )
                    loss_dict["loss"] = loss_dict["loss"] + contrastive_weight * c_loss
                    loss_dict["action_loss"] = c_loss

            loss = loss_dict["loss"] / args.grad_accum
            loss.backward()
            epoch_loss += loss_dict["loss"].item()
            epoch_steps += 1

            if (batch_idx + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(all_trainable, max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

        # Flush remaining
        if epoch_steps % args.grad_accum != 0:
            torch.nn.utils.clip_grad_norm_(all_trainable, max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            global_step += 1

        elapsed = time.time() - t_start
        avg_loss = epoch_loss / max(epoch_steps, 1)
        lr_now = scheduler.get_last_lr()[0]
        print(f"  Epoch {epoch + 1}/{args.epochs}: loss={avg_loss:.4f}, lr={lr_now:.2e}, "
              f"time={elapsed:.1f}s, steps={global_step}")

        if (epoch + 1) % args.save_every == 0 or (epoch + 1) == args.epochs:
            ckpt_path = ckpt_dir / f"world_model_s2_epoch{epoch + 1}.pt"
            torch.save({
                "epoch": epoch + 1,
                "global_step": global_step,
                "temporal_state_dict": model.temporal_layers.state_dict(),
                "action_injections_state_dict": model.action_injections.state_dict(),
                "action_encoder_state_dict": model.action_encoder.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": {
                    "temporal_every_n": args.temporal_every_n,
                    "num_frames": args.num_frames,
                    "resolution": args.resolution,
                    "stage": 2,
                },
            }, ckpt_path)
            print(f"  Saved: {ckpt_path}")

    final_path = ckpt_dir / "world_model_s2_final.pt"
    torch.save({
        "epoch": args.epochs,
        "global_step": global_step,
        "temporal_state_dict": model.temporal_layers.state_dict(),
        "action_injections_state_dict": model.action_injections.state_dict(),
        "action_encoder_state_dict": model.action_encoder.state_dict(),
        "config": {"stage": 2, "num_frames": args.num_frames, "resolution": args.resolution},
    }, final_path)
    print(f"\nTraining complete! Final: {final_path}")


def main():
    parser = argparse.ArgumentParser(description="Stage 2: Action Conditioning for Z-Image World Model")
    parser.add_argument("--model_path", type=str, default="weights/Z-Image-Turbo")
    parser.add_argument("--stage1_checkpoint", type=str,
                        default="checkpoints/zimage_stage1_v2/world_model_final.pt")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory with action-labeled video data")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--num_frames", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--lr_temporal", type=float, default=None,
                        help="LR for temporal layers when --unfreeze_temporal (defaults to --lr)")
    parser.add_argument("--contrastive_weight", type=float, default=0.0,
                        help="Weight for action contrastive loss (requires batch_size>=2)")
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--temporal_every_n", type=int, default=3)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/zimage_stage2")
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--unfreeze_temporal", action="store_true",
                        help="Unfreeze temporal layers for joint training")
    args = parser.parse_args()
    if args.lr_temporal is None:
        args.lr_temporal = args.lr
    train_stage2(args)


if __name__ == "__main__":
    main()
