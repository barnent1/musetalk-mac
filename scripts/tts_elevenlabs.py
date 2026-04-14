"""Generate a 16 kHz mono WAV from ElevenLabs TTS.

MuseTalk's audio_processor resamples via librosa to 16 kHz, so any sample rate
works in practice, but we write a true 16 kHz mono file to avoid surprises.
"""
import argparse
import os
import sys
from pathlib import Path

import requests


def load_env(env_path: Path) -> dict:
    env = {}
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip()
    return env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True, help="Text for TTS")
    parser.add_argument("--out", required=True, help="Output WAV path")
    parser.add_argument("--voice-id", default=None, help="Override voice id")
    parser.add_argument("--model", default="eleven_multilingual_v2")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    env = {**load_env(project_root / ".env"), **os.environ}

    api_key = env.get("ELEVENLABS_API_KEY")
    voice_id = args.voice_id or env.get("ELEVENLABS_VOICE_ID")
    if not api_key or not voice_id:
        sys.exit("Missing ELEVENLABS_API_KEY or ELEVENLABS_VOICE_ID")

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    params = {"output_format": "pcm_16000"}  # 16 kHz, 16-bit PCM
    headers = {"xi-api-key": api_key, "Content-Type": "application/json"}
    payload = {
        "text": args.text,
        "model_id": args.model,
        "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
    }

    print(f"Requesting ElevenLabs TTS (voice={voice_id}, {len(args.text)} chars)...")
    r = requests.post(url, params=params, headers=headers, json=payload, timeout=60)
    if r.status_code != 200:
        sys.exit(f"ElevenLabs error {r.status_code}: {r.text[:500]}")

    pcm_bytes = r.content
    # Wrap raw PCM in a WAV container.
    import wave
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out_path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)  # 16-bit
        w.setframerate(16000)
        w.writeframes(pcm_bytes)
    print(f"Wrote {out_path} ({len(pcm_bytes)/32000:.1f}s)")


if __name__ == "__main__":
    main()
