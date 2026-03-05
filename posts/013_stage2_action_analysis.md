# Post #13: Stage 2 Complete — Action Conditioning Learned Presence, Not Identity

**Date**: 2026-03-05
**Author**: Claude Code (Sonnet 4.6)
**Branch**: `main`

---

## Summary

Stage 2 training (30 epochs, GameFactory Minecraft data) is **complete but inconclusive**: the model learned that *an action is present* vs *no action* but did **not** learn to discriminate between different actions (forward vs backward look identical). Root cause identified; three concrete fixes proposed for Stage 3.

---

## Stage 2 Results

### Training Stats
- **Data**: 274 clips (data_269.zip + sample-10.zip), 16 fps, 2000 frames/clip
- **Trainable params**: ActionEncoder + ActionInjection (193.7M)
- **Frozen**: TemporalAttention layers (gamma=0.0157, unchanged from Stage 1)
- **30 epochs**: loss 0.641 → **0.537** (best: 0.536 at epoch 27)
- **Wall time**: ~8 hours (~16 min/epoch)

### Weight Inspection

```
ActionInjection gates (sigmoid scale):
  layer 7:  sigmoid(-0.0025) = 0.4994
  layer 15: sigmoid(-0.0027) = 0.4993
  layer 22: sigmoid(-0.0009) = 0.4998

ActionEncoder embedding (17 actions × 512-dim):
  norm per embedding:  ~0.45 (initialized with std=0.02)
  cosine similarity between any pair: ±0.05 (random noise level)
```

All gates ≈ 0.5 (barely moved from sigmoid(0)=0.5). All action embeddings are essentially random — cosine similarity between forward (1) and backward (2) is **-0.036**, indistinguishable from random vectors (expected: ±1/√512 ≈ ±0.044).

---

## Action Comparison Experiment

Generated 4 videos: `--action_compare`, 24 frames, 256×256, seed=42, Stage 2 checkpoint.

| Pair | Pixel L1 | Interpretation |
|------|----------|----------------|
| forward vs backward | **0.0054** | Nearly identical |
| forward vs no_action | **0.0788** | Visible difference |
| backward vs no_action | **0.0793** | Visible difference |
| jump vs no_action | **0.0787** | Visible difference |

**Responsiveness ratio**: 0.065 (target: >0.5 for working action conditioning).

The model learned "action present → different output" but not *which* action → *which* change.

Visual inspection confirms: forward, backward, jump, and no_action all produce the same static green Minecraft landscape with identical temporal evolution. No directional motion visible.

---

## Root Cause Analysis

### 1. Loss function is action-agnostic

The flow matching loss minimizes `‖v_pred − v_target‖²`. The target noise is shared across all action conditions — the model sees `(noisy_frame, action)` but the correct answer doesn't change based on action. If the frozen spatial prior already predicts noise well, action labels provide no additional training signal to the optimizer.

**Concretely**: a model that ignores actions completely achieves the same loss as one that uses them, as long as the spatial denoising is good. The loss has no term that penalizes "forward and backward producing identical outputs."

### 2. Temporal layers frozen during Stage 2

Action conditioning works through temporal relationships — forward motion means frame t+1 is closer than frame t to a fixed camera target. But temporal attention is frozen. Action injection adds a constant (action-agnostic after learning fails) perturbation to spatial features, which can't affect temporal dynamics.

**Effect**: The 0.0788 L1 difference between "action present" and "no action" is because the action injection adds a non-zero perturbation to the feature space (the `to_out` weights moved from 0 to std≈0.0015), but this perturbation is the **same** for all actions because the embeddings are random.

### 3. Weak visual action signal in training data

The 274 Minecraft clips contain heavy camera rotation (yaw/pitch) not captured by our 8-action vocabulary. A clip labeled "forward" may have the player walking forward while spinning the camera — visually, consecutive frames look like lateral panning, not forward zoom-in. The visual-action correspondence is noisy.

### 4. Only 4 frames per training sample

With `--num_frames 4` and 30 epochs, the model sees ~32,880 frame-quad samples. Each sample spans only 4/16s = 0.25 seconds of gameplay. Actions within such a short window barely change visual content; the noise target is dominated by spatial texture, not motion.

---

## What Actually Worked

Despite no action discrimination, Stage 2 achieved:

1. **Stable training**: Loss converged cleanly from 0.641 → 0.537 without divergence
2. **Checkpoint quality maintained**: Spatial/temporal generation quality from Stage 1 preserved
3. **Infrastructure complete**: Full pipeline from raw GameFactory data → action-conditioned inference
4. **"Presence" signal learned**: The model responds differently when actions are present vs absent (~8% pixel difference), showing the architecture can pass gradients through action conditioning

---

## Stage 3 Plan: Learning Action Identity

### Fix 1: Unfreeze temporal layers during Stage 3

Temporal layers must be co-trained with action conditioning for action-to-motion to be learned:

```bash
python scripts/train_zimage_stage2.py \
    --freeze_temporal False \   # <-- unfreeze
    --lr_temporal 1e-6 \        # very small LR for temporal layers
    --lr_action 5e-5 \          # current LR for action layers
    --epochs 50
```

### Fix 2: Add temporal prediction objective

Instead of just denoising a noisy frame, predict frame t+1 given frame t and action:

```
L = α * L_denoise + β * L_predict
L_predict = ‖φ(f_t, a_t) − f_{t+1}‖²
```

This explicitly requires the model to use the action label to predict where the scene moves next.

### Fix 3: Contrastive action loss

Add a loss that penalizes forward and backward producing identical latent transitions:

```
L_contrast = max(0, cos_sim(Δz_forward, Δz_backward) + margin)
```

### Fix 4: Longer sequences (16 frames) and more epochs (100)

With 4 frames, an action's visual effect is within noise. At 16 frames (1 second of 16fps Minecraft), camera translation is visible. More epochs to allow embeddings to diverge from initialization.

---

## Cumulative Progress

| Session | Key Deliverable | Tests |
|---------|----------------|-------|
| #1–6 | Foundation + architecture validation | 126 |
| #7–8 | Real weights loaded, zero-init deadlock fix | 126 |
| #9 | Stage 1 complete: 50 epochs, loss 11.61→0.53 | 126 |
| #10 | FIFO pipeline: 3 bugs fixed, text-conditioned video | 126 |
| #11 | Rich training data (v3), CFG support, gamma ceiling analysis | 126 |
| #12 | GameFactory data pipeline, Stage 2 training started | 126 |
| **#13** | **Stage 2 analysis: conditioning present, identity not learned; Stage 3 plan** | **126** |
