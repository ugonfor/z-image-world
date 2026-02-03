# Z-Image World Model

An interactive "world model" system similar to Google DeepMind's Genie 3, using open-source components. The system accepts image input (current frame) and keyboard input (user actions) to generate the next frame in real-time.

## Features

- **Real-time Frame Generation**: Generate game-like environments at 20+ FPS
- **Action-Conditioned Generation**: 17 discrete actions (movement, camera, interactions)
- **Streaming Inference**: Rolling KV cache with sink tokens for temporal consistency
- **Motion-Aware Noise Control**: Adaptive noise scheduling based on motion estimation
- **LoRA Fine-tuning**: Parameter-efficient training for action conditioning

## Architecture

```
┌─────────────┐     ┌─────────────┐
│ Current     │     │  Keyboard   │
│ Frame       │     │  Input      │
└──────┬──────┘     └──────┬──────┘
       │                   │
       ▼                   ▼
┌─────────────┐     ┌─────────────┐
│ VAE Encode  │     │ Action      │
│ (Stream-VAE)│     │ Encoder     │
└──────┬──────┘     └──────┬──────┘
       │                   │
       └───────┬───────────┘
               ▼
       ┌───────────────┐
       │ Modified DiT  │◄──── Rolling KV Cache
       │ (Causal S3-DiT│      + Sink Tokens
       │  + LoRA)      │
       └───────┬───────┘
               │
               ▼
       ┌───────────────┐
       │ Stream-VAE    │
       │ Decode        │
       └───────┬───────┘
               │
               ▼
       ┌───────────────┐
       │ Next Frame    │
       │ (Output)      │
       └───────────────┘
```

## Installation

```bash
# Clone the repository
git clone https://github.com/your-repo/z-iamge-world.git
cd z-iamge-world

# Create conda environment
conda create -n z-image-world python=3.11
conda activate z-image-world

# Install dependencies
pip install -r requirements.txt

# Install Z-Image (from diffusers)
pip install git+https://github.com/huggingface/diffusers.git
```

## Quick Start

### Interactive Demo

```bash
# Run demo with a trained model
python -m demo.interactive_app --model checkpoints/final.pt --image initial.png

# Run demo without a model (for testing)
python -m demo.interactive_app
```

### Training

#### Stage 1: Causal Adaptation

```python
from training import DiffusionForcingTrainer, DiffusionForcingConfig
from models import CausalDiT
from data import VideoOnlyDataset, create_dataloader

# Initialize model
model = CausalDiT.from_pretrained("THUDM/Z-Image-Omni-Base")

# Create dataset
dataset = VideoOnlyDataset("data/videos", num_frames=8)
dataloader = create_dataloader(dataset, batch_size=4)

# Create trainer
config = DiffusionForcingConfig()
trainer = DiffusionForcingTrainer(model, vae, config, optimizer)

# Train
for epoch in range(100):
    trainer.train_epoch(dataloader)
```

#### Stage 2: Action Fine-tuning

```python
from training import ActionFinetuner, ActionFinetuneConfig
from models import ActionEncoder

# Initialize action encoder
action_encoder = ActionEncoder(num_actions=17)

# Create fine-tuner (applies LoRA)
config = ActionFinetuneConfig()
finetuner = ActionFinetuner(model, action_encoder, vae, config)

# Train
for epoch in range(50):
    finetuner.train_epoch(dataloader)
```

### Inference

```python
from inference import RealtimePipeline, PipelineConfig
from models import ActionSpace

# Load pipeline
config = PipelineConfig()
pipeline = RealtimePipeline.from_pretrained("checkpoints/final.pt", config)

# Set initial frame
pipeline.set_initial_frame(initial_image)

# Generate frames
while True:
    action = ActionSpace.MOVE_FORWARD
    next_frame = pipeline.step(action)
    display(next_frame)
```

## Action Space

| Action | Index | Keys |
|--------|-------|------|
| Move Forward | 0 | W / ↑ |
| Move Backward | 1 | S / ↓ |
| Move Left | 2 | A / ← |
| Move Right | 3 | D / → |
| Move Forward-Left | 4 | W+A |
| Move Forward-Right | 5 | W+D |
| Move Backward-Left | 6 | S+A |
| Move Backward-Right | 7 | S+D |
| Idle | 8 | (none) |
| Look Up | 9 | I |
| Look Down | 10 | K |
| Look Left | 11 | J |
| Look Right | 12 | L |
| Jump | 13 | Space |
| Crouch | 14 | C / Ctrl |
| Interact | 15 | E |
| Attack | 16 | F |

## Project Structure

```
z-iamge-world/
├── models/
│   ├── causal_dit.py          # Modified Z-Image DiT with causal attention
│   ├── action_encoder.py      # Action embedding and injection
│   └── stream_vae.py          # Stream-VAE wrapper
├── streaming/
│   ├── rolling_kv_cache.py    # KV cache with sink tokens
│   └── motion_controller.py   # Motion-aware noise control
├── training/
│   ├── diffusion_forcing.py   # Stage 1: Causal adaptation
│   ├── action_finetune.py     # Stage 2: Action conditioning
│   └── lora_config.yaml       # LoRA configuration
├── inference/
│   ├── realtime_pipeline.py   # Main inference orchestration
│   └── input_handler.py       # Keyboard input processing
├── data/
│   ├── action_dataset.py      # Action-video pair dataloader
│   └── preprocess.py          # Frame-action alignment
├── demo/
│   └── interactive_app.py     # Interactive demo application
├── configs/
│   ├── model_config.yaml
│   └── training_config.yaml
├── requirements.txt
└── README.md
```

## Hardware Requirements

| Configuration | Specs | Expected FPS |
|--------------|-------|--------------|
| **DGX Spark (Primary)** | GB10 Superchip, 128GB unified | 20-30+ FPS |
| Development Alternative | 1x RTX 4090 (24GB), 64GB RAM | 12-18 FPS |
| Consumer Inference | RTX 4070 (12GB) | 5-8 FPS |

## Training Data

Recommended datasets for game-like environments:
1. **NVIDIA NitroGen**: Pre-annotated gamepad actions for gameplay videos
2. **OGameData (GameGen-X)**: 1M+ clips from 150+ games with annotations
3. **Custom Collection**: Screen capture + keyboard logging from target games

## References

- [Z-Image GitHub](https://github.com/Tongyi-MAI/Z-Image)
- [StreamDiffusion V2](https://github.com/chenfengxu714/StreamDiffusionV2)
- [Genie 3 Blog](https://deepmind.google/blog/genie-3-a-new-frontier-for-world-models/)
- [Vid2World Paper](https://arxiv.org/html/2505.14357v2)
- [NVIDIA NitroGen Dataset](https://huggingface.co/datasets/nvidia/NitroGen)

## License

Apache 2.0
