# z-image-world Project Memory

## Project Location
`/home/jovyan/inzoi-simulation-train--train-logit1-workspace/z-image-world/`

## Current State (as of 2026-03-08)
- **Stage 1 v3 COMPLETE**: 10 epochs, loss 0.7483→0.6756 (best), final 0.7042
  - Final checkpoint: `checkpoints/zimage_stage1_v3/world_model_final.pt`
- **Stage 2 COMPLETE** (30 epochs, GameFactory data): loss 0.641→0.537
  - Final: `checkpoints/zimage_stage2_gamefactory/world_model_s2_final.pt`
  - Responsiveness ratio: 0.065 (target > 0.5) — FAIL (action presence only, not identity)
- **Stage 3 COMPLETE** (50 epochs): loss 0.628→0.5015 (best ep38), final 0.5074
  - Checkpoints: epoch10/20/30/40/50 + final at `checkpoints/zimage_stage3/`
  - Action embeddings: still noise level (cos_sim fwd/bwd=0.027) — wrong contrastive target {0,1}
  - Gammas: 0/10 changed (bfloat16 floor) — fixed for Stage 3b
- **Stage 3b COMPLETE** (50 epochs, 2026-03-07 06:28 KST):
  - Final: `checkpoints/zimage_stage3b/world_model_s2_final.pt`, best loss 0.6060 (ep43)
  - **Action identity LEARNED**: projected (3840-dim) cos_sim(fwd/bwd) = -0.344 (was ~0)
  - Responsiveness ratio: **0.4633** (7× improvement from Stage 2's 0.065!)
  - Gammas: 10/10 changed, max delta=0.034
- **Stage 3c COMPLETE** (ep51-100, 2026-03-07 06:40 → 2026-03-08 01:51 KST): **PASS**
  - `checkpoints/zimage_stage3c/world_model_s2_epoch80.pt` — BEST checkpoint
  - **Responsiveness ratio: 0.5725** (target > 0.5 — SUCCESS!)
  - action_diff=0.0103, baseline_diff=0.0077
  - Best at ep80: projected fwd/bwd = -0.4756 (injection residual = -0.6138)
  - ep90/100 regressed due to cosine LR decay overwhelming contrastive gradient
  - Stage 3d NOT needed
- **FIFO Pipeline COMPLETE**: `inference/fifo_pipeline.py`
  - Text-conditioned video at ~1.1 gen-fps (32 steps, 256×256)
  - Action conditioning supported: `--use_actions`, `--action_pattern`, `--action_compare`
  - `scripts/run_fifo.py`: CLI for generation, inspection, and comparison
- 126/126 tests passing
- Last posts: `012_stage2_training_gamefactory.md`, `013_stage2_action_analysis.md`
- Model weights: `weights/Z-Image-Turbo/` (~31GB locally downloaded)

## Architecture Summary
- **ZImageWorldModel**: wraps Z-Image-Turbo (30 layers, 3840-dim) + TemporalAttention + ActionInjection
- **CausalDiT**: separate compact DiT for ablations (4096-dim, 28 layers)
- **Training**: DiffusionForcing (stage1) + ActionFinetune (stage2) + FlowMatching
- **Streaming**: SpatialFeatureCache (3.3x speedup by caching per-layer pre-temporal features)
- **Quantization**: INT8 dynamic quantization via torch.quantization.quantize_dynamic

## Key Files
- `models/zimage_world_model.py` - Main model with `_forward_cached()`, `_collect_spatial_features()`
- `streaming/spatial_feature_cache.py` - Context frame cache
- `inference/zimage_world_pipeline.py` - Pipeline with streaming support
- `models/quantization.py` - INT8 quantization utilities
- `models/weight_transfer.py` - Z-Image → CausalDiT weight remapping

## PyTorch Compatibility
- Running PyTorch 2.3 (nv24.03)
- `nn.RMSNorm` requires 2.4+; use `_RMSNorm` shim in zimage_world_model.py
- `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` needed for tests (set in conftest.py)
- Git remote: https://github.com/ugonfor/z-image-world.git (no auth configured)

## Data Pipelines
- **NitroGen** (`scripts/prepare_nitrogen_data.py`): downloads from YouTube — BLOCKED by IP-level bot detection; needs `--cookies` auth
- **GameFactory** (`scripts/prepare_gamefactory_data.py`): downloads from HuggingFace directly — WORKS!
  - data_269.zip (~11 GB): 269 clips at `GF-Minecraft/data_269/video/` + `metadata/` → different layout from sample
  - `_find_json()` handles both Layout A (sibling json) and Layout B (metadata/ subfolder)
- **Synthetic action** (`scripts/generate_action_synthetic.py`): fully offline, no internet needed
  - 4 scene types (corridor, outdoor, platformer, dungeon), actions drive camera motion
  - `--num_videos 500` generates training-ready data instantly
- GameFactory JSON format: `{"actions": {"0": {"ws": 0|1|2, "ad": 0|1|2, "scs": 0|1|2|3, ...}, ...}}`
  - ws=1+scs=3→RUN, ws=1→FORWARD, ws=2→BACKWARD, ad=1→LEFT, ad=2→RIGHT, scs=1→JUMP

## Remaining Work
1. **Write post #15**: Document Stage 3 journey (bugs found, fixes, Stage 3b/3c results)
2. **Plan Stage 4**: Increase absolute action_diff from 0.0103 to 0.05+ (5% pixel change)
3. **All code changes committed** (c4d7414, 74f1548, ec927de)
4. **Key bug**: `--resume` + `--epochs N` where N ≤ resume_epoch → 0 epochs trained; must use N > resume_epoch

## Stage 3c Key Lessons
- **Cosine LR decay danger**: contrastive gradient weakens as LR decays → embeddings drift back to diffusion-optimal
  - ep70-80: WORKING (projected fwd/bwd = -0.45 to -0.48)
  - ep90: regression (+0.12) — LR 3.27e-05 too low for contrastive to compete
  - ep100: partial recovery (-0.22) — oscillation at low LR
  - **Fix for Stage 4**: use cyclic LR or constant LR for contrastive training phases
- **K/V amplification mechanism**: cross-attention K/V projections learned to amplify action differences
  even without to_out growth. Injection residual improved 5x (−0.105 → −0.693) while to_out stayed flat.
- **Best checkpoint selection**: must use intermediate checkpoints (ep80), not final (ep100)

## Key Analysis Finding
- **analyze_stage3_checkpoint.py** now checks BOTH raw 512-dim AND projected 3840-dim embeddings
- Raw 512-dim embeddings look random even when projected embeddings are discriminative (projection absorbs discrimination)
- Use projected embeddings for meaningful action similarity analysis

## Diffusers Compatibility Patches (site-packages, must redo after reinstall)
- `/.local/lib/python3.10/site-packages/diffusers/utils/torch_utils.py`: handle missing `torch.xpu.*`
- `/.local/lib/python3.10/site-packages/diffusers/models/attention_dispatch.py`: `enable_gqa` TypeError fallback
- `/.local/lib/python3.10/site-packages/transformers/utils/import_utils.py`: accept nv24.xx PyTorch builds

## Key Integration Bugs Fixed
1. `nn.ModuleDict.get()` doesn't exist → use `dict[k] if k in dict else None`
2. Null caption must be 32 tokens (SEQ_MULTI_OF) not 1 to avoid RoPE pos_id mismatch
3. `cap_feat_dim` must come from `transformer.config.cap_feat_dim` not hardcoded constant
4. `VideoFolderDataset`: cv2 installed without FFMPEG → switch to imageio
5. **Zero-init deadlock**: Both gamma=0 AND to_out.weight=0 → zero gradient for gamma
   - Fix: `nn.init.xavier_uniform_(self.to_out.weight, gain=0.01)` (NOT zeros)
6. **FIFO RoPE mismatch**: `_pad_with_ids` creates pos_ids of length `total_len + pad_len`
   - Fix: `cap_freqs_cis = [f[:s] for f, s in zip(cap_freqs_cis, cap_item_seqlens)]`
7. **FIFO queue index**: After shift, new tail must get `sigma_idx=0`; all others shift left
8. **FIFO text conditioning**: `_forward_with_temporal` defaulted to null captions; must pass `cap_feat_override`
9. **bfloat16 gamma freeze**: lr_temporal=5e-6 < bfloat16 minimum step ~1.2e-4 for gamma≈0.025
   - Fix A: cast gamma params to float32 at training start (auto-applied in train script)
   - Fix B: use lr_temporal ≥ 1e-4 (not 5e-5 — still too small; confirmed 0/10 gammas changed)
10. **Contrastive loss zero gradient bug**: `MSE(sim, {0,1} match)` → diff-pair target=0, init sim≈0 → grad=0
    - Fix: use `{-1,1}` match → diff-pair target=-1, grad=2*(sim+1)=2 (active from step 1)
    - File: `training/action_finetune.py` `_action_consistency_loss()`

## Stage 4 Vision (Post Stage 3c)
Current absolute pixel differences are small (~1%) — action_diff=0.0103, baseline_diff=0.0077.
The base Z-Image-Turbo (6.15B) dominates generation; action is a small perturbation.

Options for bigger improvements:
1. **Dual-forward contrastive**: Run 2 forward passes per batch (fwd+bwd actions), add loss to
   push VIDEO OUTPUTS apart. Directly trains for different videos, not just different embeddings.
   Cost: 2× GPU time per step, but most direct signal.
2. **Unfreeze base transformer**: Allow Z-Image spatial layers to adapt to action conditioning
   (currently frozen). Would need careful LR (1e-6 or lower to preserve quality).
3. **Stronger action-correlated data**: NitroGen data would have cleaner action-video pairs
   than GameFactory. Or generate synthetic action videos (already have scripts/generate_action_synthetic.py).
4. **ControlNet-style**: Feed action embedding to each transformer block via cross-attention
   (more integrated than current 3-layer injection at depths 7,15,22).
5. **Constant LR contrastive phase**: avoid cosine decay degrading embeddings — freeze diffusion
   loss and train only contrastive at constant LR for a few epochs.

Current metric context (Stage 3c ep80):
- Ratio 0.5725: action_diff=0.0103, baseline_diff=0.0077
- Target for "working" model: ratio > 0.7, absolute action_diff > 0.05 (5%)
- Current ratio > 0.5 is a proof-of-concept that action conditioning has SOME effect

## Test Command
```bash
python -m pytest tests/ -q
# Expected: 126 passed in ~2 minutes
```

## Performance Numbers (synthetic model)
- Small (76M), 1 step, 128×128: 104.7 FPS
- Small (76M), 1 step, 256×256: 103.5 FPS
- Streaming cache speedup: 3.3x for (4 ctx frames, 2 steps)
- INT8 quantization compression: ~4x for trainable layers (1965 MB → 491 MB)
