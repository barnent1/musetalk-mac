#!/bin/bash
# Start the MuseTalk FastAPI server on the Mac.
set -euo pipefail
cd "$(dirname "$0")"
source .venv/bin/activate
export PYTORCH_ENABLE_MPS_FALLBACK=1
export GLOG_minloglevel=2
exec uvicorn server:app --host 0.0.0.0 --port "${PORT:-8000}"
