#!/bin/bash
# Run MuseTalk inference on Apple Silicon (MPS).
# Usage: ./run_inference.sh [path/to/config.yaml]
set -euo pipefail

cd "$(dirname "$0")"
source .venv/bin/activate

# Let ops unsupported on MPS fall back to CPU instead of erroring.
export PYTORCH_ENABLE_MPS_FALLBACK=1
# Silence mediapipe/absl noisy INFO logs.
export GLOG_minloglevel=2

CONFIG="${1:-configs/inference/smoke.yaml}"

cd upstream
python -m scripts.inference \
  --inference_config "$CONFIG" \
  --result_dir ./results \
  --unet_model_path ./models/musetalkV15/unet.pth \
  --unet_config ./models/musetalkV15/musetalk.json \
  --whisper_dir ./models/whisper \
  --version v15 \
  --batch_size 4
