# Post #4: Real Video Training + Interactive Demo

**Date**: 2026-02-27
**Author**: Claude Code (Opus 4.6) + Codex (GPT-5.2)
**Branch**: `feature/real-video-training`

## What Was Done

### Data Preparation
- Created `scripts/prepare_data.py` for video data download + synthetic generation
- Generated 20 synthetic training videos with moving patterns
- Big Buck Bunny download support for real video data

### Training Enhancements
- Cosine LR scheduler with warmup
- Checkpoint resume (`--resume`)
- Gradient checkpointing for temporal layers (saves ~30% activation memory)
- Recursive video search in data directories
- `temporal_every_n=3` reduces trainable params from 1965M to 784M

### Training Results (Quick Test)
| Setting | Value |
|---------|-------|
| Dataset | 20 synthetic videos |
| Resolution | 128x128 |
| Temporal layers | Every 3rd (784M trainable) |
| GPU Memory | 29.4GB / 128.5GB |
| Speed | ~8s/epoch |

### Interactive Demo
- `inference/zimage_world_pipeline.py`: ZImageWorldPipeline with DDIM denoising
- `scripts/run_interactive_demo.py`: Pygame launcher with keyboard controls
- Generates initial frame via Z-Image text-to-image
- Quality presets (Q key cycles): fast/balanced/quality

### Demo Performance on DGX Spark
| Preset | Resolution | Steps | FPS |
|--------|-----------|-------|-----|
| Fast | 128x128 | 1 | **9.2** |
| Balanced | 256x256 | 2 | **2.0** |
| Quality | 384x384 | 4 | ~0.5 |

### To run the demo with display:
```bash
export DISPLAY=:0  # On DGX Spark local display
uv run python scripts/run_interactive_demo.py --quality fast
```

## Verification
- 35/35 existing tests pass
- Training runs end-to-end with video data
- Inference generates frames at 2-9 FPS depending on quality
- Pygame installed and ready for interactive use
