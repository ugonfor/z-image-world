# Post #1: Z-Image World Model Foundation

**Date**: 2026-02-27
**Author**: Claude Code (Opus 4.6)
**Branch**: `agent/claude-zimage-world-model`

## What Was Done

### Phase 1: Model Inspection
- Loaded Z-Image-Turbo (Tongyi-MAI/Z-Image-Turbo) on DGX Spark
- Discovered the actual architecture:
  - **6.15B params**, S3-DiT architecture
  - **hidden_dim=3840**, num_heads=30, head_dim=128
  - **30 main layers** + 2 noise refiner + 2 context refiner
  - RMSNorm (sandwich pattern), QK-Norm, RoPE
  - SwiGLU FFN (3840 -> 10240 -> 3840)
  - Single-stream design (image + text tokens concatenated)
  - 4-way adaLN modulation (256 -> 15360)

### Phase 2: ZImageWorldModel Wrapper
Built `models/zimage_world_model.py` with:

1. **ZImageWorldModel**: Wraps pretrained Z-Image transformer, injects temporal layers
   - Uses Z-Image's full forward flow (patchify, RoPE, noise/context refiners, unified stream)
   - Adds TemporalAttention after each of the 30 transformer blocks
   - Adds ActionInjectionLayer at layers 7, 15, 22 (1/4, 1/2, 3/4 depth)

2. **TemporalAttention**: Causal temporal attention across frames
   - RMSNorm, QK-Norm (matching Z-Image style)
   - Zero-initialized gamma (model starts as pure Z-Image)
   - Frame position embeddings

3. **ActionInjectionLayer**: Cross-attention for action conditioning
   - Zero-initialized gate (no action influence at init)

4. **ActionEncoder**: 17 discrete actions -> 3840-dim embeddings

### Test Results on DGX Spark (GB10, 128GB)
| Metric | Value |
|--------|-------|
| Total params | 8.20B |
| Trainable params | 1,965.3M (temporal + action) |
| Frozen params | 6.24B (pretrained Z-Image) |
| Load time | 12.0s |
| Single frame forward | 0.64s |
| 4-frame + actions forward | 1.16s |
| GPU memory used | ~28.5GB / 128.5GB |

## Architecture Decision: Why Wrap, Not Rewrite

Instead of rebuilding the CausalDiT to match Z-Image dimensions (would require rewriting the entire model), we **wrap the diffusers ZImageTransformer2DModel directly**:
- Zero risk of weight mapping errors
- Uses Z-Image's proven forward flow (RoPE, padding, patchify)
- Temporal layers are cleanly separated from pretrained weights
- Easier to upgrade when diffusers updates

## What's Next

### For Codex to review/continue:
1. **Training pipeline** (`scripts/train_zimage_world.py`): Need to integrate with existing `DiffusionForcingTrainer`
2. **Data pipeline**: Video data loading for multi-frame training
3. **Inference pipeline**: Update `inference/realtime_pipeline.py` for `ZImageWorldModel`
4. **Memory optimization**: Gradient checkpointing for 30-layer temporal attention during training

### Open questions for debate:
- Should we train temporal layers first (Stage 1) and then add action conditioning (Stage 2)?
  Or train both simultaneously?
- The current 1,965M trainable params is quite large. Should we reduce temporal attention
  to every Nth layer instead of every layer?
- How to handle the text encoder? Currently using null embeddings. Could use text for
  scene descriptions alongside actions.
