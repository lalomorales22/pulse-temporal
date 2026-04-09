#!/bin/bash
# PULSE temporal awareness training pipeline
#
# This script generates training data and fine-tunes a small LLM
# (Qwen 2.5 1.5B by default) with PULSE temporal awareness.
#
# Requirements:
#   pip install pulse-temporal[torch] peft trl
#
# Usage:
#   ./train.sh                          # defaults (Qwen 1.5B, 3 epochs)
#   ./train.sh --model google/gemma-3-1b-it  # use Gemma instead
#   ./train.sh --epochs 5 --lr 1e-4     # custom hyperparams

set -e

export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
EPOCHS="${EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-1}"
LR="${LR:-2e-4}"
OUTPUT="${OUTPUT:-models/pulse-temporal-lm}"
TRAIN_SIZE="${TRAIN_SIZE:-2000}"

echo "=== PULSE Temporal Training Pipeline ==="
echo "Model:      $MODEL"
echo "Epochs:     $EPOCHS"
echo "Batch size: $BATCH_SIZE"
echo "LR:         $LR"
echo "Output:     $OUTPUT"
echo ""

# Step 1: Generate training data
if [ ! -f data/temporal_train.jsonl ]; then
    echo "--- Generating training data ($TRAIN_SIZE examples) ---"
    python -m pulse_temporal.training.data_generator
else
    echo "--- Training data already exists ---"
fi

echo ""
echo "--- Starting fine-tuning ---"

# Step 2: Fine-tune
python -m pulse_temporal.training.temporal_tuner \
    --model "$MODEL" \
    --data data/temporal_train.jsonl \
    --eval data/temporal_eval.jsonl \
    --output "$OUTPUT" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --lr "$LR" \
    --lora-r 16 \
    --max-seq-length 512 \
    "$@"

echo ""
echo "=== Training complete ==="
echo "Model saved to: $OUTPUT"
echo ""
echo "To chat with your temporal-aware model:"
echo "  python examples/inference_trained.py --model $OUTPUT"
