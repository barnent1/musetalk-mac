#!/bin/bash
# Start the MuseTalk FastAPI server on the Mac.
set -euo pipefail
cd "$(dirname "$0")"
source .venv/bin/activate
export PYTORCH_ENABLE_MPS_FALLBACK=1
export GLOG_minloglevel=2

# Production defaults: CoreML VAE decoder on CPU+GPU gives ~1.6× speedup
# over PyTorch MPS with 0.02% numerical parity (visually identical).
# Opt-in flags for speed-over-quality experiments:
#   MUSETALK_TAESD=1            — swap in Tiny AutoEncoder for SD 1.5 (~22× faster
#                                 VAE but ~5-8% skin-tone shift in the blend region)
#   MUSETALK_COREML_UNET=1      — try the CoreML UNet (currently a wash due to
#                                 MPS↔CPU↔CoreML tensor shuffling per batch)
: "${MUSETALK_FP16_VAE:=1}"
: "${MUSETALK_COREML_VAE:=1}"
: "${MUSETALK_COREML_VAE_UNITS:=CPU_AND_GPU}"
export MUSETALK_FP16_VAE MUSETALK_COREML_VAE MUSETALK_COREML_VAE_UNITS
exec uvicorn server:app --host 0.0.0.0 --port "${PORT:-8000}"
