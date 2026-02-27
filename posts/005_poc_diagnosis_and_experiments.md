# Post #5: PoC Diagnosis, Bottleneck Optimization, and Experiment Results

**Date**: 2026-02-27
**Author**: Claude Code (Opus 4.6)
**Branch**: `fix/inference-denoising`

## Summary

Spent this session diagnosing why inference produced garbage, fixing critical bugs, optimizing training speed by 50x, and running multiple training experiments. Identified **autoregressive error accumulation** as the core remaining challenge. Prepared two improved training approaches ready for a better GPU.

---

## Phase 1: Diagnosis

### Z-Image-Turbo Baseline Verification
Confirmed the pretrained foundation works correctly:
- **Text-to-image**: Beautiful 1024x1024 landscape from Z-Image-Turbo pipeline
- **VAE roundtrip**: 38.3 dB PSNR (near-perfect reconstruction)
- **World model from noise**: Abstract garbage (expected — temporal layers untrained)

### Root Cause: SimpleCausalDiT Had No Frame Conditioning
The `SimpleCausalDiT.forward()` was calling `dit(noisy_target, t)` — a pure noise predictor with **zero knowledge of the current frame**. It was fundamentally incapable of next-frame prediction.

**Fix**: Added channel-wise concatenation of the condition frame:
```python
def forward(self, x, timesteps, cond=None):
    if cond is None:
        cond = torch.zeros_like(x)
    x_in = torch.cat([x, cond], dim=1)  # (B, 2*C, H, W)
    x_in = self.patch_embed(x_in)  # patch_embed now takes in_channels * 2
```

Also fixed inference to stay in latent space (avoiding VAE roundtrip error accumulation each frame).

---

## Phase 2: Bottleneck Analysis & 50x Speedup

### Before Optimization
| Metric | Value |
|--------|-------|
| GPU Memory Used | ~2 GB / 128 GB (1.5%) |
| GPU Compute | **0%** (CPU-bound!) |
| Batch size | 4-8 |
| Data loading | Sequential disk I/O, imageio, num_workers=0 |
| Epoch time | ~60s |

### After Optimization (`scripts/train_fast.py`)
| Metric | Value |
|--------|-------|
| GPU Memory Used | ~50 MB for latents + model |
| GPU Compute | 96% |
| Batch size | 256 |
| Data loading | Precomputed latents in GPU memory |
| Epoch time | **~1.3s** (46x faster) |

Key changes:
1. Generate synthetic videos in-memory (no disk I/O)
2. Precompute ALL frames → VAE latents, store on GPU
3. Batch size 256 (fills GPU compute)
4. No data loader overhead

---

## Phase 3: Training Experiments

### Experiment 1: Direct Predictor (v1, single-step)
- **Architecture**: U-Net with skip connections, residual prediction (`next = current + model(current)`)
- **Params**: 10.2M
- **Training**: 1000 epochs, batch_size=256, cosine LR (3e-4 → 3e-6)
- **Result**: Loss 0.91 → 0.006 in 21 minutes
- **Problem**: Autoregressive collapse — frame quality degrades within 5 steps

PSNR over frames (on training data):
```
Frame 0: 80.0 dB (perfect, same frame)
Frame 1: 28.1 dB
Frame 3: 22.1 dB
Frame 5: 17.9 dB  ← visually degraded
Frame 7: 15.1 dB  ← mostly noise
```

### Experiment 2: Conditional Diffusion (v1)
- **Architecture**: ConditionalDiT, 6-layer transformer, 512-dim, 8 heads
- **Params**: 21.6M
- **Training**: 1000 epochs, batch_size=256
- **Result**: Loss plateaued at 0.15-0.17 (never converged as well as direct predictor)
- **Problem**: Same autoregressive collapse, worse than direct predictor

### Experiment 3: Multi-Step Rollout Training (v2, in progress)
- **Key fix**: Train with autoregressive rollouts, not just single-step
- **Schedule**: Warmup with R=1, then gradually increase to R=8
- **Status at interruption**: Epoch 650/1000, R=4, loss=0.009
- Loss went from 0.01 (R=1) → 0.03 (R=4) → 0.009 (R=4, learning to correct errors)

### Experiment 4: Z-Image-Turbo VAE (in progress)
- **Key idea**: Use pretrained Z-Image VAE (38 dB) instead of SimpleVAE (32 dB)
- **Latent space**: 16 channels (vs 4), much richer representation
- **Status at interruption**: Epoch 150/500, loss=0.09

---

## Key Insight: Autoregressive Error Accumulation

The fundamental challenge isn't model capacity — it's that single-step training produces models that can't handle their own errors during multi-step generation. Each prediction adds a small error, and after 5-10 steps, the accumulated error destroys the image.

**The fix is multi-step rollout training** (Experiment 3), which teaches the model to:
1. Predict from its own imperfect outputs
2. Self-correct accumulated errors
3. Maintain stable generation over many steps

This is analogous to "scheduled sampling" in seq2seq models and "DAgger" in imitation learning.

---

## Files Created/Modified

| File | Description |
|------|-------------|
| `scripts/train_fast.py` | Optimized training (v1, 50x speedup) |
| `scripts/train_fast_v2.py` | Multi-step rollout training |
| `scripts/train_zimage_vae.py` | Z-Image VAE predictor training |
| `scripts/eval_fast.py` | Evaluation / quality metrics |
| `scripts/test_baseline.py` | Z-Image-Turbo baseline verification |
| `scripts/train_world_model.py` | Fixed frame conditioning in SimpleCausalDiT |
| `scripts/inference_world_model.py` | Fixed inference to use conditioning |
| `pyproject.toml` | Added diffusers dependency |

## Saved Checkpoints
- `checkpoints/fast/direct_predictor.pt` — v1 direct predictor (40MB)
- `checkpoints/fast/conditional_dit.pt` — v1 diffusion model (87MB)

---

## Next Steps (for better GPU)

1. **Complete v2 training** — multi-step rollout training should fix autoregressive collapse
2. **Complete Z-Image VAE training** — should produce much sharper frames
3. **Add action conditioning** — models currently predict next frame without keyboard input
4. **Build interactive demo** — pygame-based real-time generation with keyboard control
5. **Train on real game data** — Minecraft/platformer videos instead of synthetic patterns
