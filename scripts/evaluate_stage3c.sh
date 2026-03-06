#!/bin/bash
# Evaluate Stage 3c final checkpoint: action comparison + responsiveness measurement
# Usage: bash scripts/evaluate_stage3c.sh [checkpoint_path]
#
# Auto-selects best checkpoint from zimage_stage3c/ if not provided.

set -e
cd /home/jovyan/inzoi-simulation-train--train-logit1-workspace/z-image-world

CKPT="${1:-checkpoints/zimage_stage3c/world_model_s2_final.pt}"

# If final checkpoint is still epoch 50 (start-of-training copy), use latest epoch ckpt
EPOCH=$(PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python -c "
import torch
ck = torch.load('$CKPT', map_location='cpu', weights_only=False)
print(ck.get('epoch', 0))
" 2>/dev/null)

if [ "$EPOCH" -le 50 ] 2>/dev/null; then
    # Find the latest epoch checkpoint
    LATEST=$(ls -t checkpoints/zimage_stage3c/world_model_s2_epoch*.pt 2>/dev/null | head -1)
    if [ -n "$LATEST" ]; then
        echo "Final checkpoint is epoch $EPOCH (startup copy); using $LATEST"
        CKPT="$LATEST"
    fi
fi

echo "=== Stage 3c Evaluation ==="
echo "Checkpoint: $CKPT"
echo "Date: $(date)"

OUT_DIR="outputs/action_compare_stage3c"
mkdir -p "$OUT_DIR"

# 1. Checkpoint analysis
echo ""
echo "=== Step 1: Checkpoint Analysis ==="
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python scripts/analyze_stage3_checkpoint.py \
    --checkpoint "$CKPT" \
    --baseline checkpoints/zimage_stage3b/world_model_s2_final.pt \
    2>&1 | tee "$OUT_DIR/analysis.log"

# 2. Action comparison videos
echo ""
echo "=== Step 2: Action Comparison Videos ==="
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python scripts/run_fifo.py \
    --checkpoint "$CKPT" \
    --action_compare \
    --output "$OUT_DIR/action_compare.gif" \
    --use_actions \
    --num_frames 24 \
    --height 256 --width 256 \
    --seed 42 \
    2>&1 | tee "$OUT_DIR/inference.log"

# 3. Responsiveness measurement
echo ""
echo "=== Step 3: Responsiveness Measurement ==="
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python scripts/measure_action_responsiveness.py \
    --forward  "$OUT_DIR/action_compare_forward.gif" \
    --backward "$OUT_DIR/action_compare_backward.gif" \
    --no_action "$OUT_DIR/action_compare_no_action.gif" \
    --jump "$OUT_DIR/action_compare_jump.gif" \
    2>&1 | tee "$OUT_DIR/responsiveness.log"

echo ""
echo "=== COMPLETE ==="
echo "Results in: $OUT_DIR/"
