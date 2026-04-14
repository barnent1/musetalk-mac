# MuseTalk on Apple Silicon

**Native-Mac port of [Tencent's MuseTalk](https://github.com/TMElyralab/MuseTalk) — realtime-quality lip-synced talking heads running entirely on Apple M-series GPUs. No CUDA, no cloud GPU bill for inference.**

MuseTalk is a SOTA audio-driven lip-sync model. The upstream release targets NVIDIA GPUs via CUDA and depends on the OpenMMLab stack (`mmpose`, `mmcv`, `mmdet`) which is notoriously painful on Apple Silicon. This repo ports the full inference pipeline to run natively on Apple M-series with Metal Performance Shaders (MPS), swaps the problem dependencies for Mac-friendly equivalents, and wraps it in an HTTP server that's a drop-in replacement for cloud GPU lipsync endpoints.

If you've got a Mac Studio sitting around, this turns it into a production-grade talking-head server.

---

## What works today

- ✅ **Full MuseTalk v1.5 inference on Apple M-series** (MPS backend)
- ✅ **Mediapipe face detection** — replaces `mmpose` / DWPose / OpenMMLab (which don't build cleanly on arm64)
- ✅ **FastAPI server** exposing the same JSON contract as the production/cloud GPU endpoints (`POST /`, `/lipsync_stream`, `/warmup`) — point your existing app at `http://mac.local:8000` and it just works
- ✅ **ElevenLabs TTS + Kling (Replicate) idle generation** in the repo so you have a complete end-to-end pipeline
- ✅ **Demo web page** with live timing breakdown
- ✅ **Tailscale Funnel-ready** so a Mac behind any NAT can serve your cloud app with auto-HTTPS

### Performance (M3 Ultra, 8.3s of audio, 199 frames @ 24fps)

| Configuration | End-to-end | UNet+VAE | Notes |
|---|---|---|---|
| Baseline (naive port) | **37.4s** | 15.5s | Direct `cuda → mps` |
| + Cached face-parse masks | 27.3s | 14.9s | Don't re-run BiSeNet per output frame |
| + Pipe frames to ffmpeg (no PNG I/O) | 18.3s | 14.9s | Stream raw BGR → `ffmpeg -f rawvideo` |
| + batch_size 16 | 17.2s | 13.8s | M3 Ultra has memory to spare |
| + fp16 VAE | **17.0s** | 13.6s | fp16 UNet gave zero gain on MPS |

**4.5× faster than realtime ceiling on PyTorch/MPS.** See [the roadmap](#roadmap-to-1-2-second-inference) for the path to GPU-parity latency via CoreML → Apple Neural Engine.

---

## Quickstart

```bash
# 1. Clone
git clone https://github.com/barnent1/musetalk-mac.git
cd musetalk-mac

# 2. Python 3.11 venv + Mac-friendly deps
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements-mac.txt

# 3. Model weights (~4 GB, one-time)
./download_weights_mac.sh

# 4. Test the inference pipeline directly
./run_inference.sh configs/inference/smoke.yaml
# → writes upstream/results/v15/yongen_yongen.mp4
```

**System requirements:** Apple Silicon (M1 / M2 / M3 / M4), macOS 14+, Python 3.11, `ffmpeg` on `PATH`.

## Run as a lipsync server

```bash
cp .env.example .env
# fill in ELEVENLABS_API_KEY and ELEVENLABS_VOICE_ID

./run_server.sh
# → FastAPI on http://0.0.0.0:8000

open http://localhost:8000     # live demo page with timing panel
```

Endpoints (same contract as cloud MuseTalk servers):

```
POST  /                  {video_b64, audio_b64, avatar_key}
                         → {video_b64, video_size_bytes, timing}

POST  /warmup            {video_b64, avatar_key}
                         → {status, timing}      # preload + cache an avatar

POST  /lipsync_stream    {avatar_key, audio_b64, video_b64?}
                         → raw mp4 bytes, X-Timing header

POST  /speak             {text, avatar_key, voice_id?}
                         → {video_b64, timing}   # ElevenLabs + lipsync in one call

GET   /health            {ok, device, cached_avatars}
```

## Expose to the public internet

If your Mac needs to serve a cloud-deployed app, the simplest path is Tailscale Funnel (free, auto-HTTPS, works behind any NAT):

```bash
tailscale funnel --bg 8000
# → https://your-hostname.tailnet-name.ts.net
```

Point your app's `MUSETALK_ENDPOINT` at that URL. Done.

---

## End-to-end pipeline (image → talking head)

```
  Portrait photo
       │
       ▼
  ┌──────────────────────────────┐
  │ Kling v2.1 (Replicate)       │  text prompt → realistic 5s idle video
  │ ↓                            │
  │ LivePortrait (Replicate)     │  locks facial identity to source photo
  └──────────────────────────────┘         ~4 min, one-time per avatar
       │
       ▼
   N idle clips (variety for conversation UI)
       │
       ▼
  ┌──────────────────────────────┐
  │  MuseTalk on this Mac  ⭐    │  audio + idle → lip-synced video
  └──────────────────────────────┘         ~17s for 8s of audio, $0 GPU cost
       │
       ▼
  Talking head video
```

Helper scripts in `scripts/`:

- `tts_elevenlabs.py` — text → 16kHz WAV
- `make_idles.py` — single portrait → N idle clips via Kling+LivePortrait on Replicate
- `make_five_idles.py` — production-style 5-clip set (2 "talking" sources + 3 idle rotation)
- `make_blink_template.py` — crafts a synthetic LivePortrait motion template so you can
  inject clean, deterministic blinks into any idle clip (Kling is unreliable about blinks)

---

## Why this is interesting

**The economics:** A single Mac Studio (~$4k one-time) serves many concurrent lipsync requests at zero per-request cost. Cloud GPU servers (A10G / A100 / H100) cost anywhere from $500–$3000/month per instance, plus the per-request billing. For low-to-mid-volume production, or for dev/staging environments, the Mac tier eliminates the ongoing spend entirely.

**The product angle:** On-device inference means a lifelike talking-head AI companion can run fully locally on a user's Mac — no data leaves the machine, no subscription, no latency spikes from provider outages. This makes applications like elder-care companions, always-on personal assistants, or offline educational tutors genuinely viable.

**The thesis:** a lot of "GPU-only" AI workloads have silently become Mac-capable. Apple Silicon's unified memory + MPS + Neural Engine is a legitimate inference platform if the code is written with it in mind, rather than naively ported.

---

## Roadmap to 1–2 second inference

The current PyTorch/MPS pipeline is bottlenecked on UNet inference (~13.6s of the 17s total). MPS lacks tensor cores and has weak fp16 acceleration, so the remaining headroom is in leaving PyTorch entirely:

| Phase | Approach | Expected end-to-end |
|---|---|---|
| ✅ A | Precompute face-parse masks, pipe ffmpeg | 18s |
| ✅ B | Batch size 16 | 17s |
| ✅ C | fp16 VAE | 17s |
| 🔜 D | CoreML UNet → Apple Neural Engine | ~5–6s |
| 🔜 E | INT8 palettization (CoreML) | ~3s |
| 🔜 F | 15fps render mode | ~2s |
| 🔜 G | MLX rewrite of remaining hot paths | ~1.5s |
| 🔜 H | Streaming output (first frame in <1s regardless of total) | — |

`ml-stable-diffusion` has already proven the SD 1.5 UNet → CoreML → ANE path; MuseTalk's UNet shares that architecture (with a 384-dim audio cross-attention instead of 768-dim text). PRs welcome.

---

## What's inside

```
musetalk-mac/
├── server.py                  # FastAPI server (drop-in for cloud MuseTalk)
├── demo.html                  # web UI with live timing breakdown
├── run_server.sh              # starts the server on :8000
├── run_inference.sh           # CLI inference for testing
├── download_weights_mac.sh    # fetches ~4GB of model weights
├── requirements-mac.txt       # Mac-friendly Python deps (no tensorflow, no openmmlab)
├── scripts/
│   ├── tts_elevenlabs.py
│   ├── make_idles.py
│   ├── make_five_idles.py
│   └── make_blink_template.py
└── upstream/                  # MuseTalk source with minimal surgical patches
    ├── musetalk/
    │   ├── models/            # UNet, VAE, PositionalEncoding (patched for MPS)
    │   └── utils/
    │       ├── preprocessing.py    # ⭐ rewritten — mediapipe instead of mmpose
    │       ├── face_parsing/       # patched for MPS device
    │       ├── audio_processor.py
    │       └── blending.py
    └── scripts/inference.py   # CLI; patched for MPS
```

### Key changes vs upstream MuseTalk

- `upstream/musetalk/utils/preprocessing.py` — rewritten from scratch. Uses mediapipe
  Face Mesh instead of DWPose (`mmpose` + `mmcv` + `mmdet`), eliminating the entire
  OpenMMLab stack. ~150 lines, behaviorally equivalent for MuseTalk's use case.
- Device selection everywhere: `cuda → mps → cpu` fallback chain
- `torch.load(..., weights_only=False)` for PyTorch 2.6+ compatibility with legacy weights
- Mediapipe pinned to `0.10.14` — newer versions removed the `solutions` API
- TensorFlow dropped — training-only dep, not needed for inference
- `numpy` bumped to 1.26.x for Apple Silicon wheel availability

---

## License & credits

MIT License — same as upstream MuseTalk.

Built on the excellent work by:
- **[MuseTalk](https://github.com/TMElyralab/MuseTalk)** (Tencent Music Entertainment Lyra Lab) — the core model
- **[MediaPipe](https://github.com/google/mediapipe)** (Google) — face mesh detection
- **[LivePortrait](https://github.com/KwaiVGI/LivePortrait)** (Kuaishou) — companion idle generation (Replicate-only currently)
- **[Diffusers](https://github.com/huggingface/diffusers)** (Hugging Face) — VAE + UNet scaffolding

If you build something with this, I'd love to hear about it. Open an issue or PR.
