# Agents Guide for z-image-world

## Project Overview
Interactive "World Model" system (similar to Google DeepMind's Genie 3).
Takes image input (current frame) + keyboard input (user action) and generates the next frame in real-time.

## Tech Stack
- Python 3.12, PyTorch 2.10 (CUDA 13.0)
- Package manager: `uv`
- Target hardware: DGX Spark (GB10, 128GB unified memory)

## Project Structure
```
z-image-world/
├── models/          # Core model architectures (CausalDiT, VAE, ActionEncoder)
├── training/        # Training logic (DiffusionForcing, ActionFinetuner)
├── inference/       # Real-time inference pipeline
├── streaming/       # KV cache, motion controller
├── data/            # Dataset classes and preprocessing
├── scripts/         # Entry points (train, inference, benchmark)
├── demo/            # Interactive demos (pygame, Z-Image)
├── tests/           # 35 test cases
└── configs/         # YAML configurations
```

## Running Commands
```bash
# Install dependencies
uv sync --extra dev --extra video

# Run tests
uv run python -m pytest tests/ -v

# Train world model
uv run python scripts/train_world_model.py --download --quick

# Run inference
uv run python scripts/inference_world_model.py --num_frames 30 --save_gif
```

## Current Status (99% complete)
- All core functionality is implemented and working
- 35/35 tests passing
- One remaining TODO: `models/causal_dit.py` line 604 - `from_pretrained` weight loading

## Collaboration Rules
1. Work on your own branch (e.g., `agent/codex-*` or `agent/claude-*`)
2. Create merge requests when work is done
3. Debate constructively with the other AI agent
4. Write progress posts after completing each task chapter
