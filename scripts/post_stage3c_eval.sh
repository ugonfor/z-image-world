#!/bin/bash
# Post-Stage 3c evaluation using BEST checkpoint (ep80, not ep100)
# Run after training completes. Uses ep80 which had the best action discrimination.
# Supersedes auto-launcher's evaluate_stage3c.sh call.
set -e
cd /home/jovyan/inzoi-simulation-train--train-logit1-workspace/z-image-world

BEST_CKPT="checkpoints/zimage_stage3c/world_model_s2_epoch80.pt"
OUT_DIR="outputs/action_compare_stage3c_best"

echo "=== Post-Stage 3c Evaluation (Best Checkpoint: ep80) ==="
echo "Date: $(date)"
echo "Note: ep80 was best — ep90 showed projected fwd/bwd regression (-0.4756 → +0.1219)"
echo "      due to cosine LR decay killing contrastive gradient"
mkdir -p "$OUT_DIR"

# 1. Action comparison videos
echo ""
echo "=== Step 1: Action Comparison Videos (ep80) ==="
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python scripts/run_fifo.py \
    --checkpoint "$BEST_CKPT" \
    --action_compare \
    --output "$OUT_DIR/action_compare.gif" \
    --use_actions \
    --num_frames 24 \
    --height 256 --width 256 \
    --seed 42 \
    2>&1 | tee "$OUT_DIR/inference.log"

# 2. Responsiveness measurement
echo ""
echo "=== Step 2: Responsiveness Measurement ==="
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python scripts/measure_action_responsiveness.py \
    --forward  "$OUT_DIR/action_compare_forward.gif" \
    --backward "$OUT_DIR/action_compare_backward.gif" \
    --no_action "$OUT_DIR/action_compare_no_action.gif" \
    --jump "$OUT_DIR/action_compare_jump.gif" \
    2>&1 | tee "$OUT_DIR/responsiveness.log"

RATIO=$(grep "responsiveness_ratio:" "$OUT_DIR/responsiveness.log" | tail -1 | awk '{print $2}')
echo ""
echo "=== RESULT ==="
echo "  Best checkpoint: ep80"
echo "  responsiveness_ratio: $RATIO"
echo "  Stage 3b baseline: 0.463"
echo "  Target: 0.5"

# Launch Stage 3d from ep80 (with injection contrastive loss)
NEEDS_3D=$(python3 -c "print('yes' if float('${RATIO:-0}') < 0.5 else 'no')" 2>/dev/null)
echo "  Needs Stage 3d: $NEEDS_3D"

if [ "$NEEDS_3D" = "yes" ]; then
    echo ""
    echo "=== Launching Stage 3d from ep80 ==="
    PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python scripts/train_zimage_stage2.py \
        --model_path weights/Z-Image-Turbo \
        --stage1_checkpoint "$BEST_CKPT" \
        --resume "$BEST_CKPT" \
        --data_dir data/videos/gamefactory \
        --checkpoint_dir checkpoints/zimage_stage3d \
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
else
    echo ""
    echo "=== Stage 3c SUCCESS with ep80 checkpoint! ==="
    echo "  No Stage 3d needed."
fi
