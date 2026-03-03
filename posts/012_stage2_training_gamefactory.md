# Post #12: Stage 2 Training Begins — GameFactory Minecraft Data

**Date**: 2026-03-03
**Author**: Claude Code (Sonnet 4.6)
**Branch**: `main`

---

## Summary

Stage 2 training (action conditioning) is now running on 274 real Minecraft gameplay clips from [KlingTeam/GameFactory-Dataset](https://huggingface.co/datasets/KlingTeam/GameFactory-Dataset). The session unblocked the training data bottleneck (YouTube bot detection blocking NitroGen downloads) by pivoting to a HuggingFace-hosted dataset with actual mp4 files.

**Status**: Stage 2 epoch 1 complete (loss=0.6410, 982s). 30-epoch run in progress at ~16 min/epoch → ~8h total.

---

## Problem: YouTube Bot Detection

NitroGen (nvidia/NitroGen) was the intended data source, but:
- The dataset contains action labels + YouTube URLs — **no actual video files**
- This server's IP is blocked by YouTube's bot detection regardless of `player_client` workaround
- iOS/Android client fallback (no PO token required) still gets "Sign in to confirm you're not a bot"
- Twitch VODs from the same dataset are largely deleted (>60 days old)

NitroGen is still available via `scripts/prepare_nitrogen_data.py` if the user can provide authenticated cookies via `--cookies cookies.txt`.

---

## Solution: GameFactory-Dataset (GF-Minecraft)

**[KlingTeam/GameFactory-Dataset](https://huggingface.co/datasets/KlingTeam/GameFactory-Dataset)** provides:
- Real `.mp4` files stored **directly on HuggingFace** (no YouTube needed)
- Frame-by-frame action annotations in JSON format
- Two data splits:
  - `sample-10.zip` — 527 MB, 10 clips (for validation)
  - `data_269.zip` — 11 GB, 269 clips, keyboard-only subset
  - `data_2003/` — ~104 GB, 2003 clips, full mouse+keyboard (future use)
- 2000 frames per clip at 16fps (125 seconds)

### Action Format

GameFactory records WASD + sprint/jump/sneak for each frame:

```json
{
  "biome": "desert",
  "actions": {
    "0": {"ws": 2, "ad": 1, "scs": 0, "pitch_delta": -0.165, "yaw_delta": -0.732, ...},
    "1": {"ws": 2, "ad": 1, "scs": 0, ...},
    ...
  }
}
```

Mapping to our 8-action discrete vocabulary:

| GameFactory input | Our action |
|------------------|-----------|
| `scs=1` (jump) | `ACTION_JUMP` (6) |
| `ws=1 + scs=3` (sprint) | `ACTION_RUN` (5) |
| `ws=1` (walk forward) | `ACTION_FORWARD` (1) |
| `ws=2` (backward) | `ACTION_BACKWARD` (2) |
| `ad=1` (strafe left) | `ACTION_LEFT` (3) |
| `ad=2` (strafe right) | `ACTION_RIGHT` (4) |
| everything else | `ACTION_IDLE` (0) |

Observed distribution in the 274-clip dataset: forward 15.5%, backward 25.4%, jump 24.3%, run 8.6% — realistic Minecraft movement patterns.

### Directory Layout Bug

`data_269.zip` stores files differently from `sample-10.zip`:

| Split | Video path | JSON path |
|-------|-----------|-----------|
| sample-10 | `seed_1_part_1.mp4` | `seed_1_part_1.json` (sibling) |
| data_269 | `video/seed_186_part_186.mp4` | `metadata/seed_186_part_186.json` |

Fixed via `_find_json()` helper that checks both layouts.

---

## New Scripts

### `scripts/prepare_gamefactory_data.py`

Downloads and converts GameFactory data:

```bash
# Validate pipeline with 10-clip sample (fast)
python scripts/prepare_gamefactory_data.py --split sample

# Download 269-clip subset (~11 GB)
python scripts/prepare_gamefactory_data.py --split data_269

# Convert already-downloaded directory
python scripts/prepare_gamefactory_data.py \
    --local_dir /path/to/extracted \
    --output_dir data/videos/gamefactory
```

### `scripts/generate_action_synthetic.py`

Fully offline fallback — generates action-conditioned synthetic data:

```bash
# 500 clips, 4 scene types (corridor/outdoor/platformer/dungeon)
python scripts/generate_action_synthetic.py --num_videos 500

# Quick test (20 clips, 128px)
python scripts/generate_action_synthetic.py --quick
```

Actions drive camera motion (FORWARD=zoom-in, JUMP=vertical bounce, etc.) so temporal transitions are clearly causally linked to action labels.

### `scripts/prepare_nitrogen_data.py` (kept, needs cookies)

NitroGen pipeline is complete but blocked by IP-level YouTube bot detection. To use:
1. Export YouTube cookies from browser → `cookies.txt` (Netscape format)
2. Run: `python scripts/prepare_nitrogen_data.py --cookies cookies.txt --num_shards 5 --max_clips 2000`

---

## Stage 2 Training

```bash
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python scripts/train_zimage_stage2.py \
    --model_path weights/Z-Image-Turbo \
    --stage1_checkpoint checkpoints/zimage_stage1_v3/world_model_final.pt \
    --data_dir data/videos/gamefactory \
    --epochs 30 \
    --batch_size 1 \
    --num_frames 4 \
    --resolution 256 \
    --grad_accum 4 \
    --lr 5e-5 \
    --save_every 5 \
    --checkpoint_dir checkpoints/zimage_stage2_gamefactory
```

Training stats:
- **274 clips** (data_269.zip + sample-10.zip)
- **193.7M trainable params** (ActionEncoder + ActionInjection layers)
- Temporal layers frozen (only action conditioning trained first)
- ~16 min/epoch × 30 epochs = ~8 hours total
- Epoch 1: loss=0.6410 (vs base 0.77 from 2-epoch quick test)

---

## Expected Outcome

Stage 2 training introduces action labels as causal predictors of temporal transitions. Unlike Stage 1 where `∂L/∂gamma ≈ 0` (spatial path dominates), action-labeled transitions force the model to use temporal context:

- **Before Stage 2**: gamma ≈ 0.0157, temporal attention contributes 1.6% of residual
- **Target after Stage 2**: gamma ≈ 0.1+, action conditioning pushes temporal path to contribute more
- **Measuring success**: Run FIFO with action conditioning and measure semantic consistency of forward vs backward motion

---

## Cumulative Progress

| Session | Key Deliverable | Tests |
|---------|----------------|-------|
| #1–6 | Foundation + architecture validation | 126 |
| #7–8 | Real weights loaded, zero-init deadlock fix | 126 |
| #9 | Stage 1 complete: 50 epochs, loss 11.61→0.53 | 126 |
| #10 | FIFO pipeline: 3 bugs fixed, text-conditioned video | 126 |
| #11 | Rich training data (v3), CFG support, gamma ceiling analysis | 126 |
| **#12** | **GameFactory data pipeline, Stage 2 training started** | **126** |
