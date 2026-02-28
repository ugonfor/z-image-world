# Claude Session Notes — For Next Session Continuity

**Last updated**: 2026-02-27
**Last branch**: `fix/inference-denoising`

## Project Overview
Interactive world model: image + keyboard → next frame. Built on Z-Image-Turbo (6.15B params).
User is supervisor. Claude works autonomously, writes posts after chapters, can use Codex coworker.

## Current State
- Z-Image-Turbo baseline verified working (text-to-image, VAE roundtrip 38.3 dB)
- SimpleVAE trained (32.6 dB PSNR, 1.3M params) — checkpoint at `checkpoints/world_model/vae.pt`
- Direct predictor v1 trained (loss 0.006) — checkpoint at `checkpoints/fast/direct_predictor.pt`
- Diffusion v1 trained (loss 0.15) — checkpoint at `checkpoints/fast/conditional_dit.pt`
- **Both v1 models suffer from autoregressive collapse** (frames degrade after 5 steps)
- v2 multi-step rollout training was in progress (epoch 650/1000, R=4, loss 0.009) — interrupted, no checkpoint saved
- Z-Image VAE predictor training was in progress (epoch 150/500, loss 0.09) — interrupted, no checkpoint saved

## Key Technical Findings

### Critical Bug Fixed
`SimpleCausalDiT.forward()` had NO frame conditioning — it was `dit(noisy_target, t)` with zero knowledge of current frame. Fixed by concatenating condition frame along channel dim.

### Training Bottleneck Fixed
Training was CPU-bound (0% GPU, sequential disk I/O). Fixed with precomputed GPU latents → **50x speedup** (60s/epoch → 1.3s/epoch).

### Core Challenge: Autoregressive Error Accumulation
Models trained on single-step prediction (frame[t]→frame[t+1]) collapse during multi-step generation. Small per-step errors compound: PSNR drops ~3dB/frame.

**Fix**: Multi-step rollout training (`train_fast_v2.py`):
- Warmup phase: single-step (R=1)
- Then progressively increase rollout length to R=8
- Model learns to predict from its own imperfect outputs
- Analogous to scheduled sampling / DAgger

### Z-Image VAE >> SimpleVAE
Z-Image-Turbo's pretrained VAE (16ch, 38dB) produces much sharper reconstructions than our SimpleVAE (4ch, 32dB). Script `train_zimage_vae.py` uses it.

## Key Scripts
| Script | Purpose | Status |
|--------|---------|--------|
| `scripts/train_fast.py` | v1 optimized training (direct + diffusion) | Done, checkpoints saved |
| `scripts/train_fast_v2.py` | v2 multi-step rollout training | Ready to run from scratch |
| `scripts/train_zimage_vae.py` | Z-Image VAE predictor | Ready to run from scratch |
| `scripts/eval_fast.py` | Evaluation with PSNR metrics | Ready |
| `scripts/test_baseline.py` | Z-Image-Turbo baseline verification | Done |
| `scripts/train_world_model.py` | Simple pipeline (fixed conditioning) | Done |

## How to Resume

### Option A: Run v2 multi-step rollout (uses SimpleVAE, needs `checkpoints/world_model/vae.pt`)
```bash
python scripts/train_fast_v2.py --num_videos 100 --num_frames 16 --epochs 1000 \
  --batch_size 128 --max_rollout 8 --warmup_epochs 200 --generate_frames 15
```

### Option B: Run Z-Image VAE predictor (downloads Z-Image-Turbo VAE automatically)
```bash
python scripts/train_zimage_vae.py --num_videos 100 --num_frames 16 --resolution 256 \
  --epochs 500 --batch_size 64 --max_rollout 8 --warmup_epochs 100 --generate_frames 15
```

Both scripts are self-contained: generate synthetic data → encode to latents → train → evaluate.

## Next Steps (Priority Order)
1. Complete v2 or Z-Image VAE training — fix autoregressive collapse
2. Add action conditioning — models currently ignore keyboard input
3. Build interactive demo — pygame real-time generation
4. Train on real game data (Minecraft/platformer)

## User Preferences
- Autonomous work, don't ask questions unless necessary
- Write posts to `posts/` after major chapters
- Priority: PoC demo → real game data → research artifact → deployable product

## Posts Written
1. `posts/001_zimage_world_model_foundation.md`
2. `posts/002_codex_review.md`
3. `posts/003_training_inference_complete.md`
4. `posts/004_training_and_demo.md`
5. `posts/005_poc_diagnosis_and_experiments.md` ← this session
