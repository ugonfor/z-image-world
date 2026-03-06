#!/bin/bash
# Stage 3b: Fixed action identity training
#
# Key fixes vs Stage 3:
#   1. Contrastive loss: target {-1,1} instead of {0,1} → active gradients for ALL pairs
#   2. Gamma float32 cast: bfloat16 floor fix (auto-applied by training script)
#   3. lr_temporal=1e-4: 20x increase (5e-6 → 1e-4) for reliable gamma updates
#   4. contrastive_weight=0.3: 3x increase to amplify action identity signal
#   5. Resume from Stage 3 final checkpoint (epoch 50 temporal weights)
#
# Expected: gammas will move (changing from Stage 2 values), embeddings will diverge
# to opposite-sign similarities for different actions within 10-20 epochs.

set -e

STAGE3_CKPT="checkpoints/zimage_stage3/world_model_s2_final.pt"
OUT_DIR="checkpoints/zimage_stage3b"

echo "Waiting for Stage 3 final checkpoint: $STAGE3_CKPT"
while [ ! -f "$STAGE3_CKPT" ]; do
    sleep 300
    echo "  $(date '+%H:%M:%S'): still waiting..."
done
echo "Stage 3 complete! Launching Stage 3b..."

PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python scripts/train_zimage_stage2.py \
    --model_path weights/Z-Image-Turbo \
    --stage1_checkpoint "$STAGE3_CKPT" \
    --data_dir data/videos/gamefactory \
    --checkpoint_dir "$OUT_DIR" \
    --epochs 50 \
    --batch_size 2 \
    --num_frames 8 \
    --resolution 256 \
    --grad_accum 2 \
    --lr 5e-5 \
    --lr_temporal 1e-4 \
    --contrastive_weight 0.3 \
    --unfreeze_temporal \
    --save_every 5 \
    2>&1 | tee logs/train_stage3b.log
