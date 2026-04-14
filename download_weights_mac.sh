#!/bin/bash
# Download MuseTalk inference weights for macOS / Apple Silicon.
# Skips dwpose (replaced by mediapipe) and syncnet (training only).
set -euo pipefail

cd "$(dirname "$0")/upstream"
CheckpointsDir="models"
mkdir -p \
  $CheckpointsDir/musetalkV15 \
  $CheckpointsDir/face-parse-bisent \
  $CheckpointsDir/sd-vae \
  $CheckpointsDir/whisper

# MuseTalk V1.5 UNet weights
huggingface-cli download TMElyralab/MuseTalk \
  --local-dir $CheckpointsDir \
  --include "musetalkV15/musetalk.json" "musetalkV15/unet.pth"

# SD-VAE
huggingface-cli download stabilityai/sd-vae-ft-mse \
  --local-dir $CheckpointsDir/sd-vae \
  --include "config.json" "diffusion_pytorch_model.bin"

# Whisper-tiny (audio encoder)
huggingface-cli download openai/whisper-tiny \
  --local-dir $CheckpointsDir/whisper \
  --include "config.json" "pytorch_model.bin" "preprocessor_config.json"

# Face parsing (BiSeNet)
gdown 154JgKpzCPW82qINcVieuPH3fZ2e0P812 \
  -O $CheckpointsDir/face-parse-bisent/79999_iter.pth
curl -L https://download.pytorch.org/models/resnet18-5c106cde.pth \
  -o $CheckpointsDir/face-parse-bisent/resnet18-5c106cde.pth

echo "✅ Weights downloaded to upstream/models/"
