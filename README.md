# Z-Image World Model

Google DeepMind의 Genie 3와 유사한 인터랙티브 "월드 모델" 시스템입니다.
이미지 입력(현재 프레임)과 키보드 입력(사용자 액션)을 받아 다음 프레임을 실시간으로 생성합니다.

## 현재 상태

| 기능 | 상태 | 설명 |
|------|------|------|
| **World Model 학습** | ✅ 작동 | 비디오 데이터로 VAE + DiT 학습 |
| **World Model 추론** | ✅ 작동 | 30+ FPS 프레임 생성 |
| Z-Image 이미지 생성 | ✅ 작동 | Tongyi-MAI/Z-Image-Turbo 사용 |
| 벤치마크 | ✅ 작동 | 160+ FPS (dummy model) |
| 테스트 | ✅ 작동 | 35개 테스트 통과 |
| 실시간 인터랙티브 | ⚠️ 제한적 | 디스플레이 필요 |

## 설치

### 1. 저장소 클론
```bash
git clone https://github.com/your-repo/z-iamge-world.git
cd z-iamge-world
```

### 2. uv로 환경 설정 (권장)
```bash
# uv 설치 (없는 경우)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 기본 의존성 설치
uv sync

# 비디오 학습용 추가 패키지
uv sync --extra video
```

### 3. 일반 pip 사용 시
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install opencv-python imageio imageio-ffmpeg
```

### DGX Spark (ARM aarch64) 사용자
pyproject.toml에 CUDA 13.0 설정이 포함되어 있어 자동으로 올바른 PyTorch가 설치됩니다.

## 빠른 시작

### 1. World Model 학습 (권장)

비디오 데이터로 World Model을 학습합니다:

```bash
# 빠른 테스트 (합성 비디오로 학습)
uv run python scripts/train_world_model.py --download --quick

# 본격 학습 (10개 비디오, 50 epoch)
uv run python scripts/train_world_model.py --download \
    --num_videos 10 \
    --vae_epochs 20 \
    --dit_epochs 50

# 커스텀 비디오 디렉토리로 학습
uv run python scripts/train_world_model.py \
    --data_dir /path/to/your/videos \
    --vae_epochs 30 \
    --dit_epochs 100
```

**학습 출력:**
```
=== Training VAE ===
  Epoch 1/20: loss=0.1264, recon=0.1264, kl=0.0617
  ...
  Epoch 20/20: loss=0.0097, recon=0.0081, kl=1.5939

=== Training Diffusion Model ===
  Epoch 1/50: loss=0.9098
  ...
  Epoch 50/50: loss=0.2153

============================================================
TRAINING COMPLETE
============================================================
Checkpoint: checkpoints/world_model/world_model.pt
```

### 2. World Model 추론

학습된 모델로 프레임을 생성합니다:

```bash
# 기본 추론 (30프레임 생성)
uv run python scripts/inference_world_model.py --num_frames 30 --save_gif

# 이미지에서 시작
uv run python scripts/inference_world_model.py \
    --input my_image.png \
    --num_frames 50 \
    --save_video

# 비디오에서 특정 프레임으로 시작
uv run python scripts/inference_world_model.py \
    --video data/videos/synthetic_video_000.mp4 \
    --frame_idx 10 \
    --num_frames 30
```

**추론 출력:**
```
Using device: cuda
Loading checkpoint from checkpoints/world_model/world_model.pt...
Model loaded successfully!

Generating 30 frames...
Generated 30 frames in 0.96s (31.25 FPS)
```

### 3. Z-Image 데모 (외부 모델)

Tongyi-MAI의 Z-Image를 사용한 고품질 이미지 생성:

```bash
uv run python demo/zimage_demo.py \
    --prompt "A cyberpunk city street at night with neon signs" \
    --num-frames 20
```

### 4. 벤치마크

```bash
uv run python scripts/benchmark.py --num_frames 100
```

## 학습 상세

### 모델 구조

| 컴포넌트 | 파라미터 | 설명 |
|----------|----------|------|
| SimpleVAE | 1.3M | 이미지 → 잠재 공간 변환 |
| SimpleCausalDiT | 21.6M | 잠재 공간에서 다음 프레임 예측 |

### 학습 파이프라인

1. **VAE 학습**: 프레임을 잠재 공간으로 인코딩/디코딩
   - 손실: MSE (재구성) + KL (정규화)

2. **DiT 학습**: 잠재 공간에서 Diffusion Forcing
   - 입력: 노이즈가 추가된 다음 프레임
   - 출력: 노이즈 예측
   - 목표: 현재 프레임에서 다음 프레임 예측

### 성능 (DGX Spark GB10)

| 단계 | 속도 |
|------|------|
| VAE 학습 | ~2초/epoch (10 videos) |
| DiT 학습 | ~3초/epoch (10 videos) |
| 추론 | **31+ FPS** |

## 프로젝트 구조

```
z-iamge-world/
├── scripts/
│   ├── train_world_model.py    # World Model 학습 스크립트 ⭐
│   ├── inference_world_model.py # World Model 추론 스크립트 ⭐
│   ├── benchmark.py            # 성능 벤치마크
│   └── test_zimage.py          # Z-Image 테스트
├── demo/
│   ├── interactive_app.py      # 인터랙티브 pygame 데모
│   └── zimage_demo.py          # Z-Image 프레임 생성 데모
├── models/
│   ├── causal_dit.py           # Causal DiT 모델
│   ├── action_encoder.py       # 액션 임베딩
│   └── stream_vae.py           # 스트리밍 VAE
├── inference/
│   └── realtime_pipeline.py    # 실시간 추론 파이프라인
├── training/
│   ├── diffusion_forcing.py    # Diffusion Forcing 학습
│   └── action_finetune.py      # 액션 조건화 학습
├── data/
│   ├── action_dataset.py       # 액션-비디오 데이터셋
│   └── videos/                 # 학습용 비디오 (자동 생성)
├── checkpoints/
│   └── world_model/            # 학습된 모델 저장
│       ├── world_model.pt      # 전체 체크포인트
│       ├── vae.pt              # VAE 가중치
│       └── dit.pt              # DiT 가중치
└── inference_output/           # 추론 결과
```

## 하드웨어 요구사항

| 환경 | 사양 | World Model 추론 |
|------|------|-----------------|
| **DGX Spark** | GB10, 128GB unified | **31+ FPS** |
| RTX 4090 | 24GB VRAM | ~40 FPS (예상) |
| RTX 4070 | 12GB VRAM | ~25 FPS (예상) |

## 커스텀 데이터로 학습

### 비디오 형식 지원
- MP4, AVI, MOV

### 디렉토리 구조
```
my_videos/
├── video_001.mp4
├── video_002.mp4
└── ...
```

### 학습 실행
```bash
uv run python scripts/train_world_model.py \
    --data_dir my_videos \
    --num_videos 100 \
    --vae_epochs 30 \
    --dit_epochs 100 \
    --resolution 256 \
    --batch_size 4
```

## 다음 단계

### 액션 조건화 학습
현재는 무조건적 프레임 예측입니다. 액션 조건화를 위해:

```python
# Stage 2: Action Fine-tuning (액션 레이블 필요)
from training import ActionFinetuner

finetuner = ActionFinetuner(model, action_encoder, vae, config)
finetuner.train_epoch(action_dataloader)
```

### 추천 데이터셋
- [NVIDIA NitroGen](https://huggingface.co/datasets/nvidia/NitroGen): 게임패드 액션 주석 포함
- [OGameData (GameGen-X)](https://github.com/GameGen-X/GameGen-X): 150+ 게임에서 1M+ 클립

## 참고 자료

- [Z-Image (Tongyi-MAI)](https://github.com/Tongyi-MAI/Z-Image) - 이미지 생성 모델
- [Genie 3 (DeepMind)](https://deepmind.google/discover/blog/genie-2-a-large-scale-foundation-world-model/) - World Model 개념
- [Diffusion Forcing](https://arxiv.org/abs/2407.01392) - 학습 방법론

## 라이선스

Apache 2.0
