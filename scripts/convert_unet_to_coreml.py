"""Convert MuseTalk V1.5 UNet to CoreML (.mlpackage) targeting Apple Neural Engine.

Shapes locked at conversion time (batch 16 for lipsync inference loop):
  sample:                [16, 8, 32, 32]  fp32
  timesteps:             []                int32 (single step, value=0)
  encoder_hidden_states: [16, 50, 384]    fp32
  → output sample:       [16, 4, 32, 32]  fp32

For smaller batch remainders we'll add a second .mlpackage with batch=1 and
pad internally when needed.
"""
import argparse
import os
import sys
import time
import warnings

import coremltools as ct
import torch
from diffusers import UNet2DConditionModel

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPSTREAM = os.path.join(ROOT, "upstream")
UNET_CFG = os.path.join(UPSTREAM, "models/musetalkV15/musetalk.json")
UNET_W = os.path.join(UPSTREAM, "models/musetalkV15/unet.pth")


class UNetWrapper(torch.nn.Module):
    """Plain tensor in/out — diffusers UNet returns a dict which coremltools doesn't like."""
    def __init__(self, unet):
        super().__init__()
        self.unet = unet

    def forward(self, sample, timesteps, encoder_hidden_states):
        return self.unet(sample, timesteps, encoder_hidden_states=encoder_hidden_states).sample


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--out", default=os.path.join(UPSTREAM, "models/musetalkV15/unet_b{batch}.mlpackage"))
    ap.add_argument("--precision", choices=["fp16", "fp32"], default="fp16")
    ap.add_argument("--compute-units", choices=["all", "cpu_and_gpu", "cpu_and_ne", "cpu_only"], default="all")
    args = ap.parse_args()

    out_path = args.out.format(batch=args.batch)

    print(f"[coreml] loading UNet from {UNET_W}")
    unet = UNet2DConditionModel.from_config(UNET_CFG)
    state = torch.load(UNET_W, map_location="cpu", weights_only=False)
    unet.load_state_dict(state)
    unet.eval()
    for p in unet.parameters():
        p.requires_grad_(False)

    wrap = UNetWrapper(unet).eval()

    # Example inputs — pass timesteps as (B,) so no broadcast/expand is needed
    # inside diffusers.Timesteps (avoids coremltools tile/expand rank error).
    B = args.batch
    sample = torch.randn(B, 8, 32, 32)
    timesteps = torch.zeros(B, dtype=torch.int32)
    encoder_hidden_states = torch.randn(B, 50, 384)

    print(f"[coreml] tracing (batch={B})…")
    t0 = time.time()
    with torch.no_grad():
        traced = torch.jit.trace(wrap, (sample, timesteps, encoder_hidden_states), strict=False)
    print(f"[coreml] traced in {time.time()-t0:.1f}s")

    precision = ct.precision.FLOAT16 if args.precision == "fp16" else ct.precision.FLOAT32
    compute_map = {
        "all": ct.ComputeUnit.ALL,
        "cpu_and_gpu": ct.ComputeUnit.CPU_AND_GPU,
        "cpu_and_ne": ct.ComputeUnit.CPU_AND_NE,
        "cpu_only": ct.ComputeUnit.CPU_ONLY,
    }
    units = compute_map[args.compute_units]

    print(f"[coreml] converting → {args.precision}, compute_units={args.compute_units}")
    t0 = time.time()
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="sample", shape=sample.shape, dtype=float),
            ct.TensorType(name="timesteps", shape=(B,), dtype=int),
            ct.TensorType(name="encoder_hidden_states", shape=encoder_hidden_states.shape, dtype=float),
        ],
        outputs=[ct.TensorType(name="predicted_sample", dtype=float)],
        convert_to="mlprogram",
        compute_precision=precision,
        compute_units=units,
        minimum_deployment_target=ct.target.macOS14,
    )
    print(f"[coreml] converted in {time.time()-t0:.1f}s")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    mlmodel.save(out_path)
    size_mb = sum(
        os.path.getsize(os.path.join(d, f))
        for d, _, files in os.walk(out_path) for f in files
    ) / (1024 * 1024)
    print(f"[coreml] wrote {out_path} ({size_mb:.0f} MB)")


if __name__ == "__main__":
    main()
