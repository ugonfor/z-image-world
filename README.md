# Z-Image World Model

Google DeepMind의 Genie 3와 유사한 인터랙티브 "월드 모델" 시스템입니다.
이미지 입력(현재 프레임)과 키보드 입력(사용자 액션)을 받아 다음 프레임을 실시간으로 생성합니다.

## 현재 상태

| 기능 | 상태 | 설명 |
|------|------|------|
| Z-Image 이미지 생성 | ✅ 작동 | Tongyi-MAI/Z-Image-Turbo 사용 |
| 액션 기반 프레임 생성 | ✅ 작동 | Img2Img + 프롬프트 기반 |
| 벤치마크 | ✅ 작동 | 160+ FPS (dummy model) |
| 테스트 | ✅ 작동 | 35개 테스트 통과 |
| 실시간 인터랙티브 | ⚠️ 제한적 | 디스플레이 필요 |
| 학습 | 🚧 준비됨 | 데이터 필요 |

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

# 의존성 설치
uv sync

# 데모용 추가 패키지
uv sync --extra demo
```

### 3. 일반 pip 사용 시
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install diffusers[torch]>=0.32.0
```

### DGX Spark (ARM aarch64) 사용자
pyproject.toml에 CUDA 13.0 설정이 포함되어 있어 자동으로 올바른 PyTorch가 설치됩니다.

## 빠른 시작

### 1. Z-Image 데모 실행 (권장)

아름다운 이미지 시퀀스를 생성합니다:

```bash
# 기본 실행 (판타지 풍경, 30프레임)
uv run python demo/zimage_demo.py

# 결과 확인
ls demo_output/
# frame_000.png ~ frame_029.png + demo.gif
```

**커스텀 프롬프트:**
```bash
uv run python demo/zimage_demo.py \
    --prompt "A cyberpunk city street at night with neon signs and flying cars" \
    --num-frames 20 \
    --output cyberpunk_demo
```

**프롬프트 예시:**
- `"A medieval castle on a cliff overlooking the ocean, dramatic lighting"`
- `"Dense jungle with ancient ruins, rays of sunlight through trees"`
- `"Snowy mountain village with cozy cabins, aurora borealis in sky"`

### 2. 테스트 실행

```bash
# 전체 테스트
uv sync --extra dev
uv run pytest tests/ -v

# 특정 테스트만
uv run pytest tests/test_models.py -v
```

### 3. 벤치마크

```bash
uv run python scripts/benchmark.py --num_frames 100
```

출력 예시:
```
============================================================
BENCHMARK RESULTS
============================================================
Throughput:
  Average FPS: 160.69
Latency (ms):
  Average: 6.22
  P95: 7.03
Memory:
  Peak GPU memory: 84.68 MB
============================================================
✓ Performance GOOD: Meets 20+ FPS target
```

### 4. 인터랙티브 데모 (디스플레이 필요)

X11 디스플레이가 있는 환경에서:
```bash
uv sync --extra demo
uv run python -m demo.interactive_app
```

**조작법:**
- `W/A/S/D` 또는 `방향키`: 이동
- `I/J/K/L`: 카메라 회전
- `Space`: 점프
- `R`: 녹화 시작/중지
- `ESC`: 종료

## 프로젝트 구조

```
z-iamge-world/
├── demo/
│   ├── interactive_app.py    # 인터랙티브 pygame 데모
│   └── zimage_demo.py        # Z-Image 프레임 생성 데모 ⭐
├── models/
│   ├── causal_dit.py         # Causal DiT 모델
│   ├── action_encoder.py     # 액션 임베딩
│   └── stream_vae.py         # 스트리밍 VAE
├── inference/
│   ├── realtime_pipeline.py  # 실시간 추론 파이프라인
│   └── input_handler.py      # 키보드 입력 처리
├── streaming/
│   ├── rolling_kv_cache.py   # KV 캐시 관리
│   └── motion_controller.py  # 모션 기반 노이즈 제어
├── training/
│   ├── diffusion_forcing.py  # Stage 1: Causal adaptation
│   └── action_finetune.py    # Stage 2: Action conditioning
├── scripts/
│   ├── benchmark.py          # 성능 벤치마크
│   └── test_zimage.py        # Z-Image 테스트
├── tests/                    # 단위 테스트
├── configs/                  # 설정 파일
└── data/                     # 데이터 로더
```

## 액션 종류

| 액션 | 인덱스 | 키 |
|------|--------|-----|
| 앞으로 | 0 | W / ↑ |
| 뒤로 | 1 | S / ↓ |
| 왼쪽 | 2 | A / ← |
| 오른쪽 | 3 | D / → |
| 대각선 이동 | 4-7 | W+A, W+D, S+A, S+D |
| 대기 | 8 | (없음) |
| 위 보기 | 9 | I |
| 아래 보기 | 10 | K |
| 왼쪽 보기 | 11 | J |
| 오른쪽 보기 | 12 | L |
| 점프 | 13 | Space |
| 앉기 | 14 | C / Ctrl |
| 상호작용 | 15 | E |
| 공격 | 16 | F |

## 하드웨어 요구사항

| 환경 | 사양 | 예상 성능 |
|------|------|----------|
| **DGX Spark** | GB10, 128GB unified | Z-Image: ~2 FPS, Benchmark: 160+ FPS |
| RTX 4090 | 24GB VRAM | Z-Image: ~3-5 FPS |
| RTX 4070 | 12GB VRAM | Z-Image: ~1-2 FPS (CPU offload 필요) |

> Z-Image 모델은 약 12GB GPU 메모리를 사용합니다.

## 다음 단계 (개발 예정)

### 실제 World Model 학습

현재 데모는 Z-Image의 Img2Img를 프롬프트로 제어합니다.
진정한 World Model을 위해서는 비디오 데이터로 학습이 필요합니다:

```python
# Stage 1: Causal Adaptation (비디오 데이터로 학습)
from training import DiffusionForcingTrainer
from data import VideoOnlyDataset

dataset = VideoOnlyDataset("data/videos", num_frames=8)
trainer = DiffusionForcingTrainer(model, vae, config, optimizer)
trainer.train_epoch(dataloader)

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
- [StreamDiffusion](https://github.com/cumulo-autumn/StreamDiffusion) - 실시간 디퓨전

## 라이선스

Apache 2.0
