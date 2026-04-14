"""Regenerate a single idle with a prompt tuned for clear full blinks."""
import asyncio
import base64
import os
import sys
import time
from pathlib import Path

import requests
import replicate

STRONG_BLINK_PROMPTS = {
    "blinky_a": (
        "A person sitting still looking directly at camera, eyes clearly closing "
        "fully and opening several times naturally throughout the clip, mouth "
        "firmly closed lips pressed together, very subtle natural micro "
        "movements, slight breathing motion, face stays centered and forward "
        "facing, calm relaxed expression, minimal head movement"
    ),
    "blinky_b": (
        "A person sitting calmly, eyes occasionally fully closing in natural "
        "deliberate blinks every few seconds, then opening wide again, mouth "
        "firmly closed lips together, gentle breathing, face stays centered, "
        "warm expression, very minimal head movement"
    ),
}
NEGATIVE = "talking, open mouth, moving lips, jaw movement, speaking, parted lips, teeth showing, large movements, sudden motion, blur, shaking, looking away, squinting, half-closed eyes"
LIVEPORTRAIT_VERSION = "067dd98cc3e5cb396c4a9efb4bba3eec6c4a9d271211325c477518fc6485e146"


def img_uri(path):
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
    print(f"[{name}] kling done {time.time()-t0:.0f}s", flush=True)
    lp = await run_lp(client, image_uri, k)
    print(f"[{name}] lp done {time.time()-t0:.0f}s", flush=True)
    path = out_dir / f"{name}.mp4"
    path.write_bytes(requests.get(lp, timeout=120).content)
    print(f"[{name}] saved {path} ({time.time()-t0:.0f}s total)", flush=True)


async def main():
    src = sys.argv[1]
    out = Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    token = os.environ["REPLICATE_API_TOKEN"]
    client = replicate.Client(api_token=token)
    image_uri = img_uri(src)
    await asyncio.gather(*[
        one(client, image_uri, name, p, out)
        for name, p in STRONG_BLINK_PROMPTS.items()
    ])


if __name__ == "__main__":
    asyncio.run(main())
