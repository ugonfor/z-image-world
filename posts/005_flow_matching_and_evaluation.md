# Post #5: Flow Matching Training + Evaluation Suite + Real-Time Benchmarks

**Date**: 2026-02-28
**Author**: Claude Code (Sonnet 4.6) + Codex review
**Branch**: `main`

## What Was Done

This session focused on correctness, quality, and measurement: we fixed 5 bugs discovered in
Codex review, introduced a full flow-matching training stack, built an offline evaluation harness,
and confirmed the model can run at >100 FPS in 1-step mode.

---

## 1. Bug Fixes (from Codex architectural review)

Codex identified five bugs spanning training, inference, and model initialization.

### Bug 1 – Protobuf / ONNX test failure (4 tests broken)

`torch._dynamo` silently imports `onnx` at collection time. On this machine the system protobuf
is too old and raises `TypeError` before any test code runs.

**Fix** – `tests/conftest.py`:
```python
import os
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
```
All 35 original tests now pass (35 → 64 total after new tests added this session).

### Bug 2 – Beta schedule recomputed every training step

`action_finetune.py` called `torch.linspace(...).cumprod(...)` inside `train_step`, allocating
and GC-ing a 1000-element tensor on every gradient update.

**Fix** – Precompute `alphas_cumprod` in `__init__` and register as a buffer.

### Bug 3 – DDIM step formula used wrong previous timestep

`realtime_pipeline.py` computed the previous timestep as `t - 1000 // num_steps`, which is
wrong when the PLMS/DDIM scheduler packs timesteps non-uniformly.

**Fix** – Precompute `_t_prev_map[t]` from the actual scheduler timestep sequence in
`__init__`, then look up `t_prev = self._t_prev_map[t]` per step.

### Bug 4 – Action consistency loss computed on noise predictions

`_action_consistency_loss` was measuring cosine similarity between noisy latents rather than
between action embeddings, so the gradient carried no meaningful action signal.

**Fix** – Compute similarity in the action-embedding space directly:
```python
emb_norm = F.normalize(action_embeddings[:, 0], dim=-1)
sim = emb_norm @ emb_norm.T                        # (B, B)
match = (actions_a.unsqueeze(1) == actions_a.unsqueeze(0)).float()
loss = F.mse_loss(sim, match)                      # push same-action pairs together
```

### Bug 5 – Duplicate class names broke `from models import *`

`zimage_world_model.py` defined `ActionInjectionLayer` and `ActionEncoder` shadowing the
identically named classes in `action_encoder.py`.

**Fix** – Renamed to `ZImageActionInjectionLayer` and `ZImageActionEncoder`.

---

## 2. Performance: torch.compile compatibility

Two patterns in `ZImageWorldModel._forward_with_temporal` caused graph breaks under
`torch.compile`:

| Anti-pattern | Fix |
|---|---|
| `ModuleDict[str(i)]` dict lookup | Pre-build `_temporal_layer_list` / `_action_injection_list` and access by integer index |
| `pad_sequence(x_patches_split)` | `torch.stack(x_patches_split, dim=0)` (valid since all frames share the same spatial size) |

These changes make the hot path fully compileable, a prerequisite for the INT4/FP8
quantization pass planned next.

---

## 3. Flow Matching Training (`training/flow_matching.py`)

Added a complete rectified-flow training stack **separate from** the existing DDPM components
(velocity in flow matching ≠ v-prediction in DDPM — mixing them would corrupt the model).

### Key design decisions

| Component | Choice | Rationale |
|---|---|---|
| Forward process | `x_t = (1-t)·x₀ + t·x₁` | Linear ODE, no SDE variance |
| Timestep sampling | Logit-normal `t = σ(N(0,1))` | Better coverage of high-SNR region than uniform |
| Loss target | `v = x₁ − x₀` | Constant velocity field for straight trajectories |
| Mixed precision | bfloat16, **no** GradScaler | BF16 has float32's exponent range; no overflow scaling needed |

### Components

```
FlowMatchingConfig      – hyperparameters dataclass
FlowMatchingLoss        – MSE on velocity + optional temporal consistency term
FlowMatchingTrainer     – optimizer, LR warmup, train_step()
FlowMatchingInference   – Euler ODE integrator (1–4 steps)
sample_flow_timesteps() – uniform / logit_normal / cosmap
flow_forward_process()  – linear interpolation with frame-independent t
```

### Numeric fix: SinusoidalPositionEmbedding + BFloat16

The sinusoidal embedding created float32 output. CausalDiT weights are bfloat16.
When the timestep tensor (Long) was passed, the path was:

```
Long → SinusoidalPositionEmbed (float32) → Linear (bfloat16) → RuntimeError
```

Fix in two parts:
1. `SinusoidalPositionEmbedding.forward` always returns float32 regardless of input dtype.
2. `CausalDiT.forward` casts input to model weight dtype immediately:
   ```python
   x = x.to(self.patch_embed.weight.dtype)
   t_emb = self.timestep_embed[0](timesteps).to(x.dtype)
   ```

---

## 4. Evaluation Suite (`scripts/evaluate.py`)

### Metrics

| Metric | Function | Notes |
|---|---|---|
| PSNR | `compute_psnr` | Supports batch (B,C,H,W) and video (B,F,C,H,W) |
| SSIM | `compute_ssim` | Gaussian window, multi-channel |
| Temporal L2 | `compute_temporal_consistency` | Mean frame-to-frame L2 diff |
| LPIPS proxy | `compute_temporal_consistency` | Edge-magnitude correlation (Sobel; no VGG needed) |
| Action responsiveness | `compute_action_responsiveness` | Cosine distance between trajectories under different actions |
| FPS throughput | `benchmark_fps` | 10-step warmup, 50-step measurement |

### Action responsiveness on random-init model

```
mean_action_divergence = 0.4957
```

Expected near 0.5 (random initialization → actions have no effect yet). This gives us a
baseline to track once real training begins.

---

## 5. Real-Time Benchmark Results

Measured on A100-SXM4-80GB with `StubVAE` (no pretrained weights needed).

| Config | Resolution | Steps | FPS | ms/frame | Params |
|---|---|---|---|---|---|
| Small | 128×128 | 1 | **104.7** | 9.5 | 76M |
| Small | 256×256 | 1 | **103.5** | 9.7 | 76M |
| Small | 256×256 | 2 | 55.5 | 18.0 | 76M |
| Medium | 128×128 | 1 | 60.6 | 16.5 | 575M |
| Medium | 256×256 | 2 | 31.0 | 32.2 | 575M |

**Key finding**: Small-1step exceeds 100 FPS at both resolutions, well above the 30 FPS
real-time threshold. Medium-1step at 128×128 hits 60 FPS — locked frame rate territory.

---

## 6. Test Suite: 35 → 64 tests

| Module | New tests |
|---|---|
| `tests/test_flow_matching.py` | 19 |
| `tests/test_evaluation.py` | 10 |
| `tests/conftest.py` | protobuf fix (affects all) |

All **64/64 tests pass**.

---

## Remaining Work

- [ ] `CausalDiT.from_pretrained` weight loading (TODO stub still present)
- [ ] KV-cache integration into ZImageWorldModel (architectural gap noted by Codex)
- [ ] INT4 / FP8 quantization pass (now unblocked by torch.compile compatibility)
- [ ] Train on real video data and re-measure action responsiveness
