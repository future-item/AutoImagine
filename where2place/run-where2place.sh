#!/bin/bash

echo "Starting placer parallel processing..."

# Parameters
DATA_JSONL="dataset/point_questions.jsonl"
IMAGE_DIR="dataset/images"
OUTPUT_DIR="output"
WORKERS=100
SESSION_ID="placer-run-$(date +%Y%m%d_%H%M%S)"
USE_MODEL="openai/gpt-4o"

echo "=== Running with baseline: ours ==="
python3 run_parallel-placer.py \
    --data_jsonl "$DATA_JSONL" \
    --image_dir "$IMAGE_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --workers $WORKERS \
    --use_session_id \
    --session_id "${SESSION_ID}-ours" \
    --use_model "$USE_MODEL" \
    --baseline ours

echo "Processing finished."