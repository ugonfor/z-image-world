# Post #14: Stage 3→3b→3c — Action Identity Learned, Responsiveness 7× Better

**Date**: 2026-03-07
**Author**: Claude Code (Sonnet 4.6)
**Branch**: `main`

---

## Summary

Stage 3 revealed two critical bugs; Stage 3b fixed them and learned discriminative action embeddings (projected cos_sim fwd/bwd = −0.344 vs ~0 before). Responsiveness ratio improved 7× (0.065 → 0.463). Stage 3c launched to further warm up action injection for ratio > 0.5.

---

## Stage 3 Epoch 20 Analysis

Stage 3 launched with three improvements over Stage 2: unfrozen temporal layers (784M trainable params vs 193.7M), contrastive loss (weight=0.1), and 8 frames per sample.

### Loss Curve

```
ep01: 0.6284  ep06: 0.5978  ep11: 0.5602  ep16: 0.5389
ep02: 0.6216  ep07: 0.5998  ep12: 0.5529  ep17: 0.5455
ep03: 0.6127  ep08: 0.5955  ep13: 0.5811  ep18: 0.5622
ep04: 0.6070  ep09: 0.5564  ep14: 0.5503  ep19: 0.5533
ep05: 0.5968  ep10: 0.5527  ep15: 0.5418  ep20: 0.5363 ★ best
```

Strong downward trend overall (0.628→0.536), with some noise. Epoch 20 is the best so far.

### Weight Inspection (Epoch 20)

```
Action embedding cosine similarities:
  cos_sim(fwd/bwd):  0.0269  (random baseline ±0.0442)
  mean off-diagonal: 0.0001 ± 0.0455
  → Still at random noise level after 20 epochs

Gates (ActionInjection):
  layer 7:  sigmoid(-0.0054) = 0.4987  (Δ=0.0029 from Stage 2)
  layer 15: sigmoid(-0.0053) = 0.4987  (Δ=0.0026 from Stage 2)
  layer 22: sigmoid(-0.0036) = 0.4991  (Δ=0.0027 from Stage 2)
  → Gates barely moving; to_out weights warming up (max 0.0149→0.0159)

Temporal gammas (10 total):
  Changed from Stage 2: 0/10  ← bitwise identical!
  Max delta: 0.00000000
```

---

## Two Critical Bugs Identified

### Bug 1: Contrastive Loss Zero Gradient for 94% of Batches

The `_action_consistency_loss` used `MSE(sim, match)` with `match ∈ {0, 1}`:
- Same-action pairs: target=1, sim≈0 → gradient = 2×(0-1) = -2 ✓ (pushes together)
- Different-action pairs: target=0, sim≈0 → gradient = 2×(0-0) ≈ 0 ✗ (**no signal**)

With batch_size=2 and 17 possible actions, probability of same-action pair ≈ 6%. So 94% of batches had **zero contrastive gradient**. Embeddings could only learn to cluster same-action pairs, never to separate different-action pairs.

**Fix** (`training/action_finetune.py`):

```python
# BEFORE: different-action → target=0 (no gradient at initialization)
match = (actions_first.unsqueeze(1) == actions_first.unsqueeze(0)).float()
loss = F.mse_loss(sim, match)

# AFTER: different-action → target=-1 (gradient=2 from initialization)
match = 2.0 * (actions_first.unsqueeze(1) == actions_first.unsqueeze(0)).float() - 1.0
mask = ~torch.eye(batch_size, dtype=torch.bool, device=sim.device)
loss = F.mse_loss(sim[mask], match[mask])
```

Effect:
- Different-pair gradient: 0.0 → 2.0 (active from step 1)
- Loss magnitude: 0.205 → 1.050 (5× larger → stronger embedding signal)

### Bug 2: bfloat16 Precision Floor Freezes Temporal Gammas

All temporal parameters (including gamma) are stored in bfloat16. Near gamma≈0.025:
- bfloat16 mantissa: 7 bits, exponent ≈ -6
- Minimum representable step: 2^(-6-7) = 2^-13 ≈ **0.000122**

AdamW effective step ≈ lr × (m/√v). With lr_temporal=5e-6 and typical Adam ratio ≈ 1:
- Effective step ≈ 5e-6 << 0.000122 → **gamma never changes in bfloat16**

Confirmed: after 1380 optimizer steps (20 epochs × 69 steps), all 10 gammas are bitwise identical to Stage 2 values.

**Fix** (`scripts/train_zimage_stage2.py`):

```python
# Cast gamma params to float32 (done automatically at training start)
for name, p in model.temporal_layers.named_parameters():
    if "gamma" in name and p.dtype == torch.bfloat16:
        p.data = p.data.float()  # float32 step at 0.025 ≈ 6e-9 (tiny, always below lr)
```

Additionally increase `lr_temporal` from 5e-6 → 1e-4 (20× increase) for Stage 3b. This ensures even gamma values near zero get meaningful updates.

---

## Stage 3b Plan

Launches automatically after Stage 3 completes (~epoch 50):

```bash
python scripts/train_zimage_stage2.py \
    --model_path weights/Z-Image-Turbo \
    --stage1_checkpoint checkpoints/zimage_stage3/world_model_s2_final.pt \
    --data_dir data/videos/gamefactory \
    --checkpoint_dir checkpoints/zimage_stage3b \
    --epochs 50 \
    --batch_size 2 \
    --num_frames 8 \
    --resolution 256 \
    --grad_accum 2 \
    --lr 5e-5 \
    --lr_temporal 1e-4 \         # 20x increase (5e-6 → 1e-4)
    --contrastive_weight 0.3 \   # 3x increase (0.1 → 0.3)
    --unfreeze_temporal \
    --save_every 5
```

Key improvements over Stage 3:
| Parameter | Stage 3 | Stage 3b |
|-----------|---------|----------|
| Contrastive target | `{0,1}` (94% zero grad) | `{-1,1}` (100% active) |
| Gamma precision | bfloat16 (frozen) | float32 (cast at start) |
| lr_temporal | 5e-6 | 1e-4 |
| contrastive_weight | 0.1 | 0.3 |

---

## Stage 3b Results

**50 epochs, resumed from Stage 3 temporal weights + fresh action weights.**

### Loss Curve
```
ep01: 0.8854  ep08: 0.7000★  ep16: 0.6400  ep22: 0.6196  ep32: 0.6285  ep43: 0.6060★★
ep02: 0.7949  ep10: 0.7161   ep17: 0.6837  ep28: 0.6595  ep34: 0.6270  ep50: 0.6374
```
Higher than Stage 3 because contrastive loss contributes ~0.10-0.15 to total.

### Key Findings: Stage 3b Final Checkpoint

**Projected (3840-dim) embeddings — the space the model actually uses:**

```
cos_sim(fwd/bwd):    -0.344  ← was ~0.027 in Stage 3 (13× more separated!)
cos_sim(idle/jump):  -0.394  ← strong separation
cos_sim(left/right): +0.845  ← grouped (both lateral strafe, visually similar)
cos_sim(fwd/run):    -0.167  ← different directions
Most dissimilar: backward & action9  = -0.931
Most similar:    action16 & left     = +0.993
Mean off-diagonal: 0.324 ± 0.514   (vs random ±0.016)
```

Raw 512-dim embeddings still show random-level similarities — the projection layers absorbed
the discrimination (correct behavior for {-1,1} contrastive loss operating on projected space).

**Temporal gammas**: 10/10 changed (max delta 0.034 from Stage 3), confirming bfloat16 fix works.

### Responsiveness Measurement

```
forward  vs backward : L1 = 0.0071  (action_diff)
forward  vs no_action: L1 = 0.0082  (baseline_diff)
responsiveness_ratio:  0.4633       (was 0.065 in Stage 2 → 7× improvement!)
```

The ratio improved from 0.065 → **0.463** — action differentiation is now comparable to
action presence. The remaining gap to 0.5 is because `to_out` weights are still small
(max=0.007 vs Stage 3's 0.015) — Stage 3b started with fresh action injection weights
and needs more warmup. Stage 3c addresses this.

### Analysis: Why `to_out` Is Small

Stage 3b used `--stage1_checkpoint` (loads only temporal weights), leaving action injection
layers freshly initialized. The `to_out` max grew from 0 → 0.007 over 50 epochs (vs Stage 3's
0.015 with 50 epochs on top of Stage 2's warmup). The discriminative embeddings are in place;
the injection pathway just needs more signal amplification.

---

## Stage 3c Plan

Continue from Stage 3b checkpoint with all weights preserved (`--resume`):

```bash
python scripts/train_zimage_stage2.py \
    --resume checkpoints/zimage_stage3b/world_model_s2_final.pt \
    --stage1_checkpoint checkpoints/zimage_stage3b/world_model_s2_final.pt \
    --epochs 100 \   # continues from epoch 50 → trains 50 more epochs
    --lr_temporal 1e-4 --contrastive_weight 0.3 --unfreeze_temporal
```

**Expected**: `to_out` max should reach 0.015+ by ep80-100, and responsiveness ratio > 0.5.

---

## What Stage 3 Did Accomplish (in hindsight)

1. **to_out weights warmed up**: max 0.002 → 0.016 (8×) — action injection pathway activated
2. **Loss hit 0.5015**: best stage so far despite buggy action discrimination
3. **Temporal dynamics**: to_out weights in temporal layers moved (total Δ=34k after 50 ep)

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
| #13 | Stage 2 analysis: conditioning present, identity not learned; Stage 3 plan | 126 |
| **#14** | **2 bugs fixed; Stage 3b: 7× responsiveness improvement (0.065→0.463); Stage 3c launched** | **126** |
