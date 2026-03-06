#!/bin/bash
# Stage 3d: Injection residual contrastive loss to break to_out stagnation
#
# Stage 3b/3c: to_out stagnates at max≈0.007 because contrastive loss trains
# only the action encoder, not the injection layer's to_out/gate weights.
# Diffusion loss alone cannot push to_out past its equilibrium.
#
# Fix: injection_contrastive_weight=1.0 directly forces to_out(cross_attn) to
# produce different outputs for different actions — trains to_out directly.
#
# Changes vs Stage 3c:
#   injection_contrastive_weight: 0.0 → 1.0 (new: trains to_out directly)
#   Resumes from Stage 3c final checkpoint

set -e
STAGE3C_CKPT="${1:-checkpoints/zimage_stage3c/world_model_s2_final.pt}"
OUT_DIR="checkpoints/zimage_stage3d"
cd /home/jovyan/inzoi-simulation-train--train-logit1-workspace/z-image-world

# Verify Stage 3c is done and has epoch >= 99
EPOCH=$(PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python -c "
import torch, sys
ck = torch.load('$STAGE3C_CKPT', map_location='cpu', weights_only=False)
e = ck.get('epoch', 0)
print(e)
sys.exit(0 if e >= 99 else 1)
" 2>/dev/null) || {
    echo "Stage 3c checkpoint not ready (epoch $EPOCH < 99). Waiting..."
    while ! PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python -c "
import torch, sys
try:
    ck = torch.load('$STAGE3C_CKPT', map_location='cpu', weights_only=False)
    sys.exit(0 if ck.get('epoch',0) >= 99 else 1)
except: sys.exit(1)
" 2>/dev/null; do
        sleep 300
        echo "  $(date '+%H:%M:%S'): still waiting..."
    done
    echo "Stage 3c complete! Launching Stage 3d..."
}

echo "=== Stage 3d: Higher contrastive weight ==="
echo "Resume from: $STAGE3C_CKPT"

PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python scripts/train_zimage_stage2.py \
    --model_path weights/Z-Image-Turbo \
    --stage1_checkpoint "$STAGE3C_CKPT" \
    --resume "$STAGE3C_CKPT" \
    --data_dir data/videos/gamefactory \
    --checkpoint_dir "$OUT_DIR" \
    --epochs 150 \
    --batch_size 2 \
    --num_frames 8 \
    --resolution 256 \
    --grad_accum 2 \
    --lr 5e-5 \
    --lr_temporal 1e-4 \
    --contrastive_weight 0.3 \
    --injection_contrastive_weight 1.0 \
    --unfreeze_temporal \
    --save_every 10 \
    2>&1 | tee logs/train_stage3d.log

echo "Stage 3d complete."
