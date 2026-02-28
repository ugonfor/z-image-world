#!/usr/bin/env python3
"""
Inspect a training checkpoint to verify temporal layers are learning.
Checks gamma values and to_out weight norms.
"""
import sys
import torch
from pathlib import Path

def inspect_checkpoint(ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    tl = ckpt["temporal_state_dict"]

    print(f"\n=== Checkpoint: {ckpt_path} ===")
    print(f"Epoch: {ckpt.get('epoch', '?')}, Step: {ckpt.get('global_step', '?')}")

    # Check gamma values
    gamma_keys = [k for k in tl.keys() if "gamma" in k]
    print(f"\n--- Gamma values ({len(gamma_keys)} params) ---")
    gamma_nonzero = 0
    for k in gamma_keys[:5]:
        v = tl[k]
        print(f"  {k}: mean={v.mean().item():.6f}, max={v.abs().max().item():.6f}, std={v.std().item():.6f}")
        if v.abs().max().item() > 1e-6:
            gamma_nonzero += 1
    if len(gamma_keys) > 5:
        more_nonzero = sum(1 for k in gamma_keys[5:] if tl[k].abs().max().item() > 1e-6)
        print(f"  ... and {len(gamma_keys) - 5} more ({more_nonzero} non-zero)")
    total_nonzero = sum(1 for k in gamma_keys if tl[k].abs().max().item() > 1e-6)
    print(f"  TOTAL: {total_nonzero}/{len(gamma_keys)} gamma params are non-zero")

    # Check to_out weights
    toout_keys = [k for k in tl.keys() if "to_out.weight" in k]
    print(f"\n--- to_out.weight norms ({len(toout_keys)} params) ---")
    for k in toout_keys[:5]:
        v = tl[k]
        print(f"  {k}: norm={v.norm().item():.6f}, max={v.abs().max().item():.6f}")

    # Check Q/K/V weights
    toq_keys = [k for k in tl.keys() if "to_q.weight" in k]
    print(f"\n--- to_q.weight norms ({len(toq_keys)} params) ---")
    for k in toq_keys[:3]:
        v = tl[k]
        print(f"  {k}: norm={v.norm().item():.4f}, max={v.abs().max().item():.6f}")

    print(f"\n--- Summary ---")
    all_nonzero = sum(1 for k, v in tl.items() if v.abs().max().item() > 1e-6)
    print(f"  Non-zero params: {all_nonzero}/{len(tl)}")
    if total_nonzero > 0:
        print("  ✓ DEADLOCK FIXED: gamma values are non-zero, learning is happening")
    else:
        print("  ✗ DEADLOCK STILL PRESENT: all gamma values are zero")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Auto-detect latest checkpoint
        ckpt_dir = Path("checkpoints/zimage_stage1_v2")
        ckpts = sorted(ckpt_dir.glob("world_model_epoch*.pt"))
        if not ckpts:
            print("No checkpoints found yet.")
            sys.exit(0)
        ckpt_path = str(ckpts[-1])
    else:
        ckpt_path = sys.argv[1]
    inspect_checkpoint(ckpt_path)
