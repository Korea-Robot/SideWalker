#!/bin/bash

# MetaUrban RL Inference Script
# Usage: ./run_inference.sh [model_path] [output_video] [max_steps]

# Default values
MODEL_PATH=${1:-"metaurban_discrete_actor_multimodal_final.pt"}
OUTPUT_VIDEO=${2:-"inference_episode.mp4"}
MAX_STEPS=${3:-1000}
DEVICE=${4:-"cuda:0"}

echo "=================================="
echo "MetaUrban RL Inference"
echo "=================================="
echo "Model path: $MODEL_PATH"
echo "Output video: $OUTPUT_VIDEO"
echo "Max steps: $MAX_STEPS"
echo "Device: $DEVICE"
echo "=================================="

# Check if model file exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model file '$MODEL_PATH' not found!"
    echo "Please check the model path and try again."
    exit 1
fi

# Create output directory if it doesn't exist
OUTPUT_DIR=$(dirname "$OUTPUT_VIDEO")
if [ "$OUTPUT_DIR" != "." ] && [ ! -d "$OUTPUT_DIR" ]; then
    echo "Creating output directory: $OUTPUT_DIR"
    mkdir -p "$OUTPUT_DIR"
fi

# Run inference
python inference.py \
    --actor_path "$MODEL_PATH" \
    --output_video "$OUTPUT_VIDEO" \
    --max_steps "$MAX_STEPS" \
    --device "$DEVICE" \
    --fps 30 \
    --seed 42

echo "=================================="
echo "Inference completed!"
echo "Video saved to: $OUTPUT_VIDEO"
echo "=================================="



# 기본 사용법
python inference.py --actor_path metaurban_discrete_actor_multimodal_final.pt

# 커스텀 설정
python inference.py \
    --actor_path checkpoints/metaurban_discrete_actor_epoch_50.pt \
    --output_video results/episode_50.mp4 \
    --max_steps 1500 \
    --device cuda:1

# 스크립트 사용 (더 편리함)
./run_inference.sh metaurban_discrete_actor_multimodal_final.pt output.mp4 1000