# Post #10: FIFO-Diffusion Pipeline — Turning Z-Image-Turbo into a Video Generator

**Date**: 2026-03-01
**Author**: Claude Code (Sonnet 4.6)
**Branch**: `main`

---

## Summary

Implemented a working FIFO-Diffusion video generation pipeline (`inference/fifo_pipeline.py`) that uses Z-Image-Turbo + temporal attention to generate coherent video sequences from text prompts. Fixed three bugs discovered during testing: queue index reset, null caption conditioning, and RoPE dimension mismatch with real text.

---

## What is FIFO-Diffusion?

FIFO-Diffusion (Kim et al., NeurIPS 2024, arXiv:2405.11473) converts any image diffusion model into a video generator:

1. Maintain a queue of K frames at staggered noise levels `[σ≈0 (clean), ..., σ_max (noisy)]`
2. Each step: forward-pass all K frames jointly through the model (temporal attention sees all frames)
3. Apply Euler step to advance each frame toward cleaner state
4. Dequeue the head (σ≈0) → output frame
5. Push new noisy frame at tail, shift queue

This makes use of the temporal attention layers added in Stage 1 training — all K frames attend to each other during each denoising step, providing temporal context.

---

## Three Bugs Fixed

### Bug 1: Queue Sigma Index Not Reset After Shift

**Symptom**: Frames 4+ degraded to artifact noise after initial clean frames.

**Root cause**: After queue shift (pop head, push new tail), `queue_step_indices` was not updated. The new tail frame's sigma index was wrong (treated as near-clean instead of pure noise).

**Fix**:
```python
# After pop/push:
queue_step_indices = torch.cat([
    queue_step_indices[1:],
    torch.zeros(1, dtype=torch.long, device=device),
])
```
Also fixed: `steps_per_frame = ceil(steps/K)` not floor, and new tail = `σ_max * noise` not a copy.

---

### Bug 2: Null Caption Conditioning (No Text Guidance)

**Symptom**: Frames denoised with no text guidance → complete noise for later frames.

**Root cause**: `_forward_with_temporal()` always used null (zero) caption embeddings regardless of the text prompt passed to the pipeline. Text conditioning was never reaching the denoiser.

**Fix**: Added `cap_feat_override` parameter threaded through `forward()` → `_forward_with_temporal()`. The pipeline now:
1. Encodes the prompt once via `ZImagePipeline.encode_prompt()` at the start of `generate()`
2. Passes the encoded features to every model forward call

```python
cap_feats, _ = self._zimage_pipe.encode_prompt(prompt, ...)
# ...
v_pred = self.model(latent_seq, t_per_frame, cap_feat_override=cap_feats)
```

---

### Bug 3: RoPE Dimension Mismatch with Real Captions

**Symptom**: `RuntimeError: The size of tensor a (32) must match tensor b (45) at non-singleton dimension 1`

**Root cause**: In `_pad_with_ids()`, caption position IDs have an off-by-one: the function creates `ori_pos_ids` with `total_padded_len` positions (e.g. 32), then ALSO appends `pad_len` more zero pad positions (e.g. 13), giving 45 total pos_ids for 32 feature tokens. The official `_prepare_sequence()` handles this with `freqs_cis[:, :feats.shape[1]]` truncation.

For a 19-token caption: `pad_len = (-19) % 32 = 13`, giving `32 + 13 = 45` pos_ids vs 32 features → exact mismatch.

Null 32-token captions weren't affected because `pad_len = 0`, so pos_ids = features = 32.

**Fix** (one line):
```python
# Truncate freqs to feature length — matches _prepare_sequence behavior
cap_freqs_cis = [f[:s] for f, s in zip(cap_freqs_cis, cap_item_seqlens)]
```

---

## Results

### Pipeline Configuration (Best)

```python
FIFOConfig(
    queue_size=8,
    num_inference_steps=32,   # 4 steps per frame per outer iteration
    height=256, width=256,
    seed_steps=6,
    anchor_init=True,          # Critical for temporal coherence
)
```

### Frame Quality

| Frame | Quality | Notes |
|-------|---------|-------|
| 0 (seed) | Excellent | Z-Image-Turbo quality — photorealistic forest, trees, path |
| 1–8 | Very Good | Clear forest structure, coherent with seed |
| 9–16 | Good | Forest structure maintained, slow color drift starting |
| 17–24 | Fair | Color drift accumulates, structural coherence reduces |

**Generation speed**: 1.13 gen-fps at 256×256 with `steps_per_frame=4`

### Anchor Initialization (Key Innovation)

New tail frames initialized with **anchor + noise** instead of pure noise:
```python
# anchor_init=True:
new_frame = last_clean + sigma_max * noise  # warm start from prev frame
# anchor_init=False:
new_frame = noise * sigma_max               # pure noise (baseline)
```

Impact: Without anchor_init, frames 9+ become complete noise/texture. With anchor_init, all 24+ frames maintain coherent forest structure.

---

## Remaining Limitation: Semantic Drift

After ~16 frames, colors shift from green (forest) toward orange/red. This is **semantic drift** — a known FIFO limitation when temporal conditioning is weak.

**Root cause**: gamma≈0.025 means temporal attention contributes only 2.5% of the residual per block. Z-Image-Turbo's spatial generation (97.5%) dominates and can pull frames toward different attractors.

**Fix**: Stage 2 training with real gameplay video should push gamma to 0.1–0.2, making temporal conditioning 4–8× stronger. With that, anchor init should produce stable long videos.

---

## Architecture Notes

```
ZImageWorldModel forward with FIFO:
1. encode_prompt() → cap_feats (once, reused for all frames)
2. Queue of 8 latents at sigma levels [0, ..., sigma_max]
3. Each FIFO step:
   - stack queue → (1, 8, 16, 32, 32) latent sequence
   - model(latent_seq, per_frame_sigmas, cap_feats=cap_feats)
     → patchify each frame, embed captions
     → noise_refiner on image tokens
     → context_refiner on caption tokens (now with real text!)
     → main transformer blocks + temporal attention + action injection
     → final layer, unpatchify
   - Euler step: x_next = x + v_pred * (σ_next - σ_t)
   - Pop clean head → output
   - Push anchor+noise tail
```

---

## What Changed This Session

1. Fixed `cap_freqs_cis` truncation bug in `models/zimage_world_model.py`
2. Added `anchor_init` to `FIFOConfig` in `inference/fifo_pipeline.py`
3. Implemented anchor initialization in queue shift logic
4. Generated `output_fifo_v5.gif` — 24-frame forest path video at 1.13 gen-fps
5. 126/126 tests still passing

---

## Next Steps

### Immediate
- **Stage 2 training**: Need `video.mp4` + `video_actions.json` pairs from inZOI gameplay
  - `scripts/train_zimage_stage2.py` is ready, just needs `--data_dir`
  - Expected improvement: gamma 0.025 → 0.1+, semantic drift eliminated

### Can Do Now
- Improve temporal coherence further with `anchor_noise_frac` (partial noise for new tail frames)
- Add classifier-free guidance (CFG) for stronger text conditioning (2× cost per step)

---

## Cumulative Progress

| Session | Key Deliverable | Tests |
|---------|----------------|-------|
| #1 | Foundation: CausalDiT, ActionEncoder, StreamVAE, RollingKVCache | 35 |
| #2–6 | Architecture validation, training pipelines, streaming, INT8 | 126 |
| #7 | Real weights downloaded, integration bugs fixed | 126 |
| #8 | Zero-init deadlock fix confirmed | 126 |
| #9 | Stage 1 complete: 50 epochs, loss 11.61→0.53 | 126 |
| **#10** | **FIFO pipeline: 3 bugs fixed, text-conditioned 24-frame video @ 1.13 fps** | **126** |
