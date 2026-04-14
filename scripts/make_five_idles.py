"""Generate 5 idle clips for one avatar:
  - 2 "talking" variants: mouth firmly closed, face centered, ready for MuseTalk
  - 3 "idle" variants: different subtle head motions for rotation between turns

Each is produced via Kling v2.1 → LivePortrait (same as production).
Then synthetic full blinks are added to the 2 talking variants.
"""
import asyncio
import base64
import os
import pickle
import sys
import time
from pathlib import Path

import requests
import replicate

# Prompts — all enforce closed mouth and subtle motion
NEGATIVE = ("talking, open mouth, moving lips, jaw movement, speaking, "
            "parted lips, teeth showing, large movements, sudden motion, "
            "blur, shaking, looking away")

TALKING_PROMPTS = [
    ("talking_a",
     "A person sitting still looking directly at camera, mouth firmly closed "
     "lips pressed together, very subtle breathing, very slight natural head "
     "movement, face centered and forward facing, calm warm expression, "
     "minimal motion"),
    ("talking_b",
     "A person sitting calmly, mouth firmly closed lips together, soft warm "
     "smile with closed lips, very slight natural head movement, face centered, "
     "attentive listening expression, minimal motion"),
]

IDLE_PROMPTS = [
    ("idle_turn_left",
     "A person sitting still, slow gentle head turn slightly to the left then "
     "back toward center, mouth firmly closed lips pressed together, gentle "
     "breathing, warm expression, subtle natural motion only"),
    ("idle_turn_right",
     "A person sitting still, slow gentle head turn slightly to the right then "
     "back toward center, mouth firmly closed lips pressed together, gentle "
     "breathing, warm expression, subtle natural motion only"),
    ("idle_nod",
     "A person sitting still, small slow nod downward then back up, mouth "
     "firmly closed lips pressed together, attentive look, gentle breathing, "
     "subtle natural motion only"),
]

LIVEPORTRAIT_VERSION = "067dd98cc3e5cb396c4a9efb4bba3eec6c4a9d271211325c477518fc6485e146"


def img_uri(path: str) -> str:
    b = open(path, "rb").read()
    ct = "image/png" if path.lower().endswith(".png") else "image/jpeg"
    return f"data:{ct};base64,{base64.b64encode(b).decode()}"


async def run_kling(client, image_uri, prompt):
    pred = await client.predictions.async_create(
        model="kwaivgi/kling-v2.1",
        input={
            "start_image": image_uri,
            "prompt": prompt,
            "negative_prompt": NEGATIVE,
            "duration": 5,
            "mode": "standard",
        },
    )
    while pred.status not in ("succeeded", "failed", "canceled"):
        await asyncio.sleep(3)
        pred = await client.predictions.async_get(pred.id)
    if pred.status != "succeeded":
        raise RuntimeError(f"Kling failed: {pred.error}")
    return pred.output if isinstance(pred.output, str) else pred.output[0]


async def run_lp(client, face_uri, driving_url):
    pred = await client.predictions.async_create(
        version=LIVEPORTRAIT_VERSION,
        input={
            "face_image": face_uri,
            "driving_video": driving_url,
            "live_portrait_dsize": 512,
            "live_portrait_scale": 2.3,
            "live_portrait_relative": True,
            "live_portrait_stitching": True,
            "live_portrait_eye_retargeting": False,
            "live_portrait_lip_retargeting": False,
        },
    )
    while pred.status not in ("succeeded", "failed", "canceled"):
        await asyncio.sleep(2)
        pred = await client.predictions.async_get(pred.id)
    if pred.status != "succeeded":
        raise RuntimeError(f"LivePortrait failed: {pred.error}")
    return pred.output if isinstance(pred.output, str) else pred.output[0]


async def one(client, image_uri, name, prompt, out_dir):
    t0 = time.time()
    print(f"[{name}] kling...", flush=True)
    k = await run_kling(client, image_uri, prompt)
    print(f"[{name}] kling {time.time()-t0:.0f}s → LP", flush=True)
    lp = await run_lp(client, image_uri, k)
    path = out_dir / f"{name}.mp4"
    path.write_bytes(requests.get(lp, timeout=120).content)
    print(f"[{name}] saved {path.name} ({time.time()-t0:.0f}s total)", flush=True)
    return path


async def main():
    src = sys.argv[1]
    out = Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)

    token = os.environ.get("REPLICATE_API_TOKEN")
    if not token:
        env = Path("/Users/barnent1/Projects/talking-companion/.env.local")
        for line in env.read_text().splitlines():
            if line.startswith("REPLICATE_API_TOKEN="):
                token = line.split("=", 1)[1].strip()
                break
    client = replicate.Client(api_token=token)
    image_uri = img_uri(src)

    jobs = TALKING_PROMPTS + IDLE_PROMPTS
    print(f"launching {len(jobs)} Replicate jobs in parallel...")
    results = await asyncio.gather(
        *[one(client, image_uri, name, p, out) for name, p in jobs],
        return_exceptions=True,
    )
    for r in results:
        if isinstance(r, Exception):
            print("  failed:", r)

    print("\n✓ all clips done in", out)


if __name__ == "__main__":
    asyncio.run(main())
