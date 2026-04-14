"""Generate idle videos for a source image using Kling v2.1 + LivePortrait
on Replicate — matching the production talking-companion pipeline.

Usage: make_idles.py SOURCE_IMAGE OUT_DIR [--n 2]
"""
import argparse
import asyncio
import base64
import os
import sys
import time
from pathlib import Path

import requests
import replicate


IDLE_PROMPTS = [
    ("idle_center", "A person sitting still looking at camera, mouth firmly closed lips pressed together, very subtle natural micro movements only, gentle slow eye blinks, very slight breathing motion, face stays centered and forward facing, calm relaxed expression, minimal head movement"),
    ("idle_tilt", "A person sitting still, slight head tilt left, soft warm smile with mouth firmly closed lips together, gentle breathing, face stays mostly centered, very subtle natural movements only"),
    ("idle_nod", "A person sitting still, small nod, attentive listening look mouth firmly closed lips together, gentle breathing, face stays mostly centered, very subtle natural movements only"),
    ("idle_smile", "A person sitting still, gentle smile with mouth firmly closed lips pressed together, warm expression, gentle breathing, face stays mostly centered, very subtle natural movements only"),
]
NEGATIVE = "talking, open mouth, moving lips, jaw movement, speaking, parted lips, teeth showing, large movements, sudden motion, blur, shaking, looking away"

LIVEPORTRAIT_VERSION = "067dd98cc3e5cb396c4a9efb4bba3eec6c4a9d271211325c477518fc6485e146"


def image_to_data_uri(path: str) -> str:
    with open(path, "rb") as f:
        b = f.read()
    ct = "image/png" if path.lower().endswith(".png") else "image/jpeg"
    return f"data:{ct};base64,{base64.b64encode(b).decode()}"


async def run_kling(client, image_uri: str, prompt: str) -> str:
    print(f"  kling start ({prompt[:40]}...)", flush=True)
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
    out = pred.output
    return out if isinstance(out, str) else out[0]


async def run_liveportrait(client, face_uri: str, driving_url: str) -> str:
    print("  liveportrait start", flush=True)
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
    out = pred.output
    return out if isinstance(out, str) else out[0]


async def make_one(client, image_uri: str, name: str, prompt: str, out_dir: Path) -> Path:
    print(f"[{name}] starting", flush=True)
    t0 = time.time()
    kling_url = await run_kling(client, image_uri, prompt)
    print(f"[{name}] kling done ({time.time()-t0:.0f}s) -> {kling_url}", flush=True)
    lp_url = await run_liveportrait(client, image_uri, kling_url)
    print(f"[{name}] lp done ({time.time()-t0:.0f}s) -> {lp_url}", flush=True)
    r = requests.get(lp_url, timeout=120)
    out_path = out_dir / f"{name}.mp4"
    out_path.write_bytes(r.content)
    print(f"[{name}] saved {out_path} ({len(r.content)/1024:.0f} KB, total {time.time()-t0:.0f}s)", flush=True)
    return out_path


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("out_dir")
    ap.add_argument("--n", type=int, default=2, help="how many idles to generate (1-4)")
    args = ap.parse_args()

    token = os.environ.get("REPLICATE_API_TOKEN")
    if not token:
        # Fall back to the talking-companion env file
        env_path = Path("/Users/barnent1/Projects/talking-companion/.env.local")
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if line.startswith("REPLICATE_API_TOKEN="):
                    token = line.split("=", 1)[1].strip()
                    break
    if not token:
        sys.exit("REPLICATE_API_TOKEN missing")
    os.environ["REPLICATE_API_TOKEN"] = token

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    image_uri = image_to_data_uri(args.source)
    client = replicate.Client(api_token=token)

    prompts = IDLE_PROMPTS[: args.n]
    results = await asyncio.gather(
        *[make_one(client, image_uri, name, prompt, out_dir) for name, prompt in prompts],
        return_exceptions=True,
    )
    failures = [r for r in results if isinstance(r, Exception)]
    print(f"\nDone. {len(results)-len(failures)}/{len(results)} succeeded.", flush=True)
    for r in failures:
        print(f"  failure: {r}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
