"""FastAPI server wrapping the Mac MuseTalk pipeline.

Matches the JSON contract of the existing Fly/GPU endpoint so the Next.js app
can point at this server unchanged:

    POST /              {video_b64, audio_b64, avatar_key?}
                        -> {video_b64, video_size_bytes, timing}
    POST /warmup        {video_b64, avatar_key}
                        -> {status, timing}
    POST /lipsync_stream {avatar_key, audio_b64, video_b64?}
                        -> raw mp4 bytes (Content-Type: application/octet-stream)
                           404 if avatar_key is not cached and no video_b64 given

Run from the repository root:
    PYTORCH_ENABLE_MPS_FALLBACK=1 \\
    .venv/bin/uvicorn server:app --host 0.0.0.0 --port 8000
"""

import base64
import copy
import glob
import os
import shutil
import subprocess
import sys
import tempfile
import time
from contextlib import asynccontextmanager

import cv2
import numpy as np
import requests
import torch
import wave
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Resolve paths relative to this file, then cd into upstream/ so the
# hard-coded "./models/..." references in the upstream code work unchanged.
ROOT = os.path.dirname(os.path.abspath(__file__))
UPSTREAM = os.path.join(ROOT, "upstream")
os.chdir(UPSTREAM)
sys.path.insert(0, UPSTREAM)

from transformers import WhisperModel  # noqa: E402

import coremltools as ct  # noqa: E402

from musetalk.utils.audio_processor import AudioProcessor  # noqa: E402
from musetalk.utils.blending import (  # noqa: E402
    get_image,
    get_image_blending,
    get_image_prepare_material,
)
from musetalk.utils.face_parsing import FaceParsing  # noqa: E402
from musetalk.utils.preprocessing import (  # noqa: E402
    coord_placeholder,
    get_landmark_and_bbox,
)
from musetalk.utils.utils import datagen, get_video_fps, load_all_model  # noqa: E402


# ─── globals populated at startup ─────────────────────────────────────────
state: dict = {}
avatar_cache: dict[str, dict] = {}


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    device = _pick_device()
    print(f"[musetalk] loading models on {device}", flush=True)
    vae, unet, pe = load_all_model(
        unet_model_path="./models/musetalkV15/unet.pth",
        vae_type="sd-vae",
        unet_config="./models/musetalkV15/musetalk.json",
        device=device,
    )
    pe = pe.to(device)
    vae.vae = vae.vae.to(device)
    unet.model = unet.model.to(device)

    # Phase C: fp16 VAE. Decoder/encoder are the same class (AutoencoderKL);
    # it auto-casts input latents to self.vae.dtype in decode_latents/encode_latents.
    use_fp16_vae = os.environ.get("MUSETALK_FP16_VAE", "1") == "1"
    if use_fp16_vae and device.type in ("mps", "cuda"):
        vae.vae = vae.vae.half()
        print("[musetalk] VAE → fp16", flush=True)

    # UNet fp16 gave no measurable gain on MPS (see Phase C profile); leave off by default.
    # Phase D: optional CoreML VAE decoder (CPU+GPU is ~1.6× faster than MPS
    # with 0.024% parity error). Requires vae_decoder_b16.mlpackage to exist.
    coreml_vae = None
    coreml_batch = int(os.environ.get("MUSETALK_COREML_BATCH", "16"))
    if os.environ.get("MUSETALK_COREML_VAE", "0") == "1":
        mlpkg = f"./models/sd-vae/vae_decoder_b{coreml_batch}.mlpackage"
        if os.path.exists(mlpkg):
            cu_name = os.environ.get("MUSETALK_COREML_VAE_UNITS", "CPU_AND_GPU")
            cu = {
                "ALL": ct.ComputeUnit.ALL,
                "CPU_AND_GPU": ct.ComputeUnit.CPU_AND_GPU,
                "CPU_AND_NE": ct.ComputeUnit.CPU_AND_NE,
                "CPU_ONLY": ct.ComputeUnit.CPU_ONLY,
            }[cu_name]
            print(f"[musetalk] loading CoreML VAE decoder ({cu_name}) from {mlpkg}…", flush=True)
            _t = time.time()
            coreml_vae = ct.models.MLModel(mlpkg, compute_units=cu)
            print(f"[musetalk] CoreML VAE loaded in {time.time()-_t:.1f}s", flush=True)
        else:
            print(f"[musetalk] MUSETALK_COREML_VAE=1 but {mlpkg} not found; falling back to PyTorch", flush=True)

    use_fp16_unet = os.environ.get("MUSETALK_FP16_UNET", "0") == "1"
    if use_fp16_unet and device.type in ("mps", "cuda"):
        unet.model = unet.model.half()
        pe = pe.half()
        print("[musetalk] UNet + PE → fp16", flush=True)

    audio_processor = AudioProcessor(feature_extractor_path="./models/whisper")
    weight_dtype = unet.model.dtype
    whisper = WhisperModel.from_pretrained("./models/whisper")
    whisper = whisper.to(device=device, dtype=weight_dtype).eval()
    whisper.requires_grad_(False)

    fp = FaceParsing(left_cheek_width=90, right_cheek_width=90)
    timesteps = torch.tensor([0], device=device)

    state.update(
        device=device,
        vae=vae,
        unet=unet,
        pe=pe,
        audio_processor=audio_processor,
        whisper=whisper,
        weight_dtype=weight_dtype,
        fp=fp,
        timesteps=timesteps,
        coreml_vae=coreml_vae,
        coreml_batch=coreml_batch,
    )
    print("[musetalk] ready", flush=True)
    yield


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Preset avatars for the local demo page — path is relative to UPSTREAM (our cwd).
AVATAR_PRESETS = {
    "demo_a": "data/demo_five/talking_a_blink.mp4",
    "demo_b": "data/demo_five/talking_b_blink.mp4",
}


def _load_env_file(path: str) -> dict:
    env = {}
    if os.path.exists(path):
        for line in open(path).read().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip()
    return env


# Load ElevenLabs creds once (env file lives at project root, one level up from UPSTREAM)
_ENV = _load_env_file(os.path.join(ROOT, ".env"))
ELEVEN_KEY = os.environ.get("ELEVENLABS_API_KEY") or _ENV.get("ELEVENLABS_API_KEY")
ELEVEN_VOICE = os.environ.get("ELEVENLABS_VOICE_ID") or _ENV.get("ELEVENLABS_VOICE_ID")


def elevenlabs_tts(text: str, voice_id: str | None = None) -> bytes:
    """Return 16 kHz mono WAV bytes for the given text."""
    if not ELEVEN_KEY:
        raise HTTPException(500, "ELEVENLABS_API_KEY not configured")
    vid = voice_id or ELEVEN_VOICE
    if not vid:
        raise HTTPException(500, "ELEVENLABS_VOICE_ID not configured")
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{vid}"
    r = requests.post(
        url,
        params={"output_format": "pcm_16000"},
        headers={"xi-api-key": ELEVEN_KEY, "Content-Type": "application/json"},
        json={
            "text": text,
            "model_id": "eleven_multilingual_v2",
            "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
        },
        timeout=60,
    )
    if r.status_code != 200:
        raise HTTPException(502, f"ElevenLabs {r.status_code}: {r.text[:200]}")
    # Wrap raw PCM into WAV
    buf = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    try:
        with wave.open(buf.name, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(r.content)
        return open(buf.name, "rb").read()
    finally:
        os.unlink(buf.name)


# ─── helpers ───────────────────────────────────────────────────────────────

EXTRA_MARGIN = 10  # v15 default


def _detect_media_kind(data: bytes) -> str:
    """Return 'image' or 'video' by inspecting magic bytes."""
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image"
    if data[:3] == b"\xff\xd8\xff" or data[6:10] in (b"JFIF", b"Exif"):
        return "image"
    return "video"


def _write_input_media(tmp: str, data: bytes) -> tuple[list[str], float]:
    """Write incoming media to tmp, return (image_list, fps)."""
    kind = _detect_media_kind(data)
    if kind == "image":
        img_path = os.path.join(tmp, "in.png")
        with open(img_path, "wb") as f:
            f.write(data)
        return [img_path], 25.0

    vid = os.path.join(tmp, "in.mp4")
    with open(vid, "wb") as f:
        f.write(data)
    frames_dir = os.path.join(tmp, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-v", "error", "-i", vid, "-start_number", "0",
         f"{frames_dir}/%08d.png"],
        check=True,
    )
    img_list = sorted(glob.glob(os.path.join(frames_dir, "*.png")))
    fps = get_video_fps(vid)
    return img_list, fps


def prepare_avatar(video_bytes: bytes) -> dict:
    """Run the expensive one-time prep: frames, landmarks, VAE encode."""
    tmp = tempfile.mkdtemp(prefix="mt_avatar_")
    try:
        img_list, fps = _write_input_media(tmp, video_bytes)
        coord_list, frame_list = get_landmark_and_bbox(img_list, 0)

        vae = state["vae"]
        fp = state["fp"]
        input_latent_list = []
        valid_coords = []
        valid_frames = []
        mask_list = []
        crop_box_list = []
        for bbox, frame in zip(coord_list, frame_list):
            if bbox == coord_placeholder:
                continue
            x1, y1, x2, y2 = bbox
            y2 = min(y2 + EXTRA_MARGIN, frame.shape[0])
            crop = frame[y1:y2, x1:x2]
            crop = cv2.resize(crop, (256, 256), interpolation=cv2.INTER_LANCZOS4)
            latents = vae.get_latents_for_unet(crop)
            input_latent_list.append(latents)
            valid_coords.append((x1, y1, x2, y2))
            valid_frames.append(frame)
            # Precompute face-parse mask + crop box for this source frame.
            # The mask depends only on the source frame, not on the generated
            # mouth — so we do this once during warmup, saving ~7s per lipsync.
            mask_array, crop_box = get_image_prepare_material(
                frame, [x1, y1, x2, y2], fp=fp, mode="jaw"
            )
            mask_list.append(mask_array)
            crop_box_list.append(crop_box)

        if not input_latent_list:
            raise HTTPException(status_code=400, detail="No face detected in input")

        return {
            "fps": fps,
            "coord_list_cycle": valid_coords + valid_coords[::-1],
            "frame_list_cycle": valid_frames + valid_frames[::-1],
            "input_latent_list_cycle": input_latent_list + input_latent_list[::-1],
            "mask_list_cycle": mask_list + mask_list[::-1],
            "crop_box_list_cycle": crop_box_list + crop_box_list[::-1],
        }
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def run_lipsync(avatar: dict, audio_bytes: bytes, batch_size: int = 16) -> bytes:
    """Given a prepared avatar and audio bytes, return final mp4 bytes."""
    prof = {}
    _t0 = time.time()

    tmp = tempfile.mkdtemp(prefix="mt_lip_")
    try:
        wav = os.path.join(tmp, "in.wav")
        with open(wav, "wb") as f:
            f.write(audio_bytes)

        device = state["device"]
        pe = state["pe"]
        unet = state["unet"]
        vae = state["vae"]
        whisper = state["whisper"]
        audio_processor = state["audio_processor"]
        weight_dtype = state["weight_dtype"]
        timesteps = state["timesteps"]
        fp = state["fp"]

        fps = avatar["fps"]
        _t = time.time()
        whisper_features, librosa_len = audio_processor.get_audio_feature(wav)
        whisper_chunks = audio_processor.get_whisper_chunk(
            whisper_features, device, weight_dtype, whisper, librosa_len,
            fps=fps, audio_padding_length_left=2, audio_padding_length_right=2,
        )
        prof["audio_ms"] = int((time.time() - _t) * 1000)

        coord_cycle = avatar["coord_list_cycle"]
        frame_cycle = avatar["frame_list_cycle"]
        latent_cycle = avatar["input_latent_list_cycle"]
        mask_cycle = avatar["mask_list_cycle"]
        crop_box_cycle = avatar["crop_box_list_cycle"]

        gen = datagen(whisper_chunks, latent_cycle, batch_size, 0, device)
        res_frame_list = []
        pe_ms = unet_ms = vae_ms = 0
        _t = time.time()
        with torch.no_grad():
            for whisper_batch, latent_batch in gen:
                _a = time.time()
                audio_feat = pe(whisper_batch)
                latent_batch = latent_batch.to(dtype=unet.model.dtype)
                if device.type == "mps":
                    torch.mps.synchronize()
                pe_ms += int((time.time() - _a) * 1000)

                _a = time.time()
                pred = unet.model(
                    latent_batch, timesteps, encoder_hidden_states=audio_feat
                ).sample
                if device.type == "mps":
                    torch.mps.synchronize()
                unet_ms += int((time.time() - _a) * 1000)

                _a = time.time()
                coreml_vae = state.get("coreml_vae")
                coreml_batch = state.get("coreml_batch", 16)
                if coreml_vae is not None and pred.shape[0] <= coreml_batch:
                    # Pad batch to fixed size expected by the mlpackage, run on
                    # CoreML (CPU+GPU), then unpack → BGR uint8.
                    B_in = pred.shape[0]
                    pred_fp32 = pred.detach().to("cpu").to(dtype=torch.float32)
                    if B_in < coreml_batch:
                        pad = torch.zeros(coreml_batch - B_in, *pred_fp32.shape[1:], dtype=torch.float32)
                        pred_fp32 = torch.cat([pred_fp32, pad], dim=0)
                    out = coreml_vae.predict({"latents": pred_fp32.numpy()})["image"]
                    out = out[:B_in]  # drop padding
                    # CoreML output: [B, 3, 256, 256] RGB in [0, 1]
                    # Convert to BGR uint8 [B, 256, 256, 3] to match vae.decode_latents return shape
                    out = np.transpose(out, (0, 2, 3, 1))  # HWC
                    out = (out * 255.0).round().clip(0, 255).astype(np.uint8)
                    recon = out[..., ::-1]  # RGB → BGR
                else:
                    recon = vae.decode_latents(pred)
                    if device.type == "mps":
                        torch.mps.synchronize()
                vae_ms += int((time.time() - _a) * 1000)

                for r in recon:
                    res_frame_list.append(r)
        prof["unet_vae_ms"] = int((time.time() - _t) * 1000)
        prof["pe_ms"] = pe_ms
        prof["unet_ms"] = unet_ms
        prof["vae_decode_ms"] = vae_ms
        prof["frames"] = len(res_frame_list)

        # Build output video by piping BGR frames directly into ffmpeg's stdin.
        # Face-parse masks are precomputed per source frame during warmup, so
        # we can use get_image_blending which is a pure compositing op.
        silent = os.path.join(tmp, "silent.mp4")
        # Frame size from first available frame
        h, w = frame_cycle[0].shape[:2]
        _t = time.time()
        ff_proc = subprocess.Popen(
            [
                "ffmpeg", "-y", "-v", "error",
                "-f", "rawvideo", "-pix_fmt", "bgr24",
                "-s", f"{w}x{h}", "-r", str(fps),
                "-i", "-",
                "-vcodec", "libx264", "-pix_fmt", "yuv420p",
                "-preset", "veryfast", "-crf", "18",
                silent,
            ],
            stdin=subprocess.PIPE,
        )
        assert ff_proc.stdin is not None
        blend_ms = 0
        pipe_ms = 0
        for i, res_frame in enumerate(res_frame_list):
            idx = i % len(coord_cycle)
            x1, y1, x2, y2 = coord_cycle[idx]
            ori = frame_cycle[idx]  # no deepcopy: get_image_blending doesn't mutate
            try:
                rs = cv2.resize(res_frame.astype(np.uint8), (x2 - x1, y2 - y1))
            except Exception:
                continue
            _t2 = time.time()
            combined = get_image_blending(
                ori, rs, [x1, y1, x2, y2],
                mask_cycle[idx], crop_box_cycle[idx],
            )
            blend_ms += int((time.time() - _t2) * 1000)
            _t3 = time.time()
            ff_proc.stdin.write(np.ascontiguousarray(combined).tobytes())
            pipe_ms += int((time.time() - _t3) * 1000)
        ff_proc.stdin.close()
        ff_proc.wait()
        prof["blend_ms"] = blend_ms
        prof["pipe_ms"] = pipe_ms
        prof["ffmpeg_encode_ms"] = int((time.time() - _t) * 1000) - blend_ms - pipe_ms

        final = os.path.join(tmp, "final.mp4")
        _t = time.time()
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-i", wav, "-i", silent,
             "-c:v", "copy", "-c:a", "aac", "-shortest", final],
            check=True,
        )
        prof["ffmpeg_mux_ms"] = int((time.time() - _t) * 1000)
        with open(final, "rb") as f:
            mp4 = f.read()
        prof["total_ms"] = int((time.time() - _t0) * 1000)
        print(f"[profile] {prof}", flush=True)
        return mp4
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ─── request models ────────────────────────────────────────────────────────


class WarmupReq(BaseModel):
    video_b64: str
    avatar_key: str


class LipsyncReq(BaseModel):
    video_b64: str | None = None
    audio_b64: str
    avatar_key: str | None = None


class StreamReq(BaseModel):
    avatar_key: str
    audio_b64: str
    video_b64: str | None = None


# ─── routes ────────────────────────────────────────────────────────────────


@app.get("/health")
def health():
    return {"ok": True, "device": str(state.get("device")), "avatars": list(avatar_cache)}


@app.post("/warmup")
def warmup(req: WarmupReq):
    start = time.time()
    video_bytes = base64.b64decode(req.video_b64)
    avatar_cache[req.avatar_key] = prepare_avatar(video_bytes)
    return {
        "status": "ready",
        "timing": {"total_s": round(time.time() - start, 3)},
    }


@app.post("/")
def lipsync(req: LipsyncReq):
    start = time.time()
    key = req.avatar_key
    if key and key in avatar_cache:
        avatar = avatar_cache[key]
    else:
        if not req.video_b64:
            raise HTTPException(status_code=400, detail="video_b64 required when avatar not cached")
        avatar = prepare_avatar(base64.b64decode(req.video_b64))
        if key:
            avatar_cache[key] = avatar

    mp4 = run_lipsync(avatar, base64.b64decode(req.audio_b64))
    return {
        "video_b64": base64.b64encode(mp4).decode(),
        "video_size_bytes": len(mp4),
        "timing": {
            "total_s": round(time.time() - start, 3),
            "gpu_cost_cents": 0,
        },
    }


class SpeakReq(BaseModel):
    text: str
    avatar_key: str = "demo_a"
    voice_id: str | None = None


@app.post("/speak")
def speak(req: SpeakReq):
    """text → ElevenLabs TTS → MuseTalk lipsync. Returns timing breakdown."""
    t_total = time.time()

    t_tts = time.time()
    wav_bytes = elevenlabs_tts(req.text, req.voice_id)
    tts_ms = int((time.time() - t_tts) * 1000)

    t_warm = time.time()
    if req.avatar_key not in avatar_cache:
        path = AVATAR_PRESETS.get(req.avatar_key)
        if not path or not os.path.exists(path):
            raise HTTPException(400, f"unknown avatar {req.avatar_key}")
        avatar_cache[req.avatar_key] = prepare_avatar(open(path, "rb").read())
    warm_ms = int((time.time() - t_warm) * 1000)

    t_lip = time.time()
    mp4 = run_lipsync(avatar_cache[req.avatar_key], wav_bytes)
    lip_ms = int((time.time() - t_lip) * 1000)

    return {
        "video_b64": base64.b64encode(mp4).decode(),
        "audio_bytes": len(wav_bytes),
        "video_bytes": len(mp4),
        "timing": {
            "tts_ms": tts_ms,
            "warmup_ms": warm_ms,
            "lipsync_ms": lip_ms,
            "total_ms": int((time.time() - t_total) * 1000),
        },
    }


app.mount("/media", StaticFiles(directory="data/demo_five"), name="media")


@app.get("/")
def home():
    return FileResponse(os.path.join(ROOT, "demo.html"))


@app.post("/lipsync")
def lipsync_alias(req: LipsyncReq):
    return lipsync(req)


@app.post("/lipsync_stream")
def lipsync_stream(req: StreamReq):
    start = time.time()
    if req.avatar_key not in avatar_cache:
        if not req.video_b64:
            raise HTTPException(
                status_code=404,
                detail=f"avatar {req.avatar_key} not cached; include video_b64 to warm up",
            )
        avatar_cache[req.avatar_key] = prepare_avatar(base64.b64decode(req.video_b64))

    mp4 = run_lipsync(avatar_cache[req.avatar_key], base64.b64decode(req.audio_b64))
    elapsed = round(time.time() - start, 3)
    return Response(
        content=mp4,
        media_type="application/octet-stream",
        headers={"X-Timing": f"total_s={elapsed}"},
    )
