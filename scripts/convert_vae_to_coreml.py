"""Convert the SD VAE decoder to CoreML .mlpackage.

Input shape is fixed at conversion to batch=16, latent 4×32×32 (the same shape
MuseTalk feeds into decode every lipsync batch).
"""
import argparse
import os
import time
import warnings

import coremltools as ct
import torch
from diffusers import AutoencoderKL

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPSTREAM = os.path.join(ROOT, "upstream")
VAE_DIR = os.path.join(UPSTREAM, "models/sd-vae")


class VAEDecoder(torch.nn.Module):
    """Wrap AutoencoderKL.decode so coremltools gets a plain tensor input/output."""
    def __init__(self, vae: AutoencoderKL, scaling_factor: float):
        super().__init__()
        self.vae = vae
        self.scaling_factor = scaling_factor

    def forward(self, latents):
        x = latents * (1.0 / self.scaling_factor)
        out = self.vae.decode(x).sample
        out = (out / 2 + 0.5).clamp(0, 1)
        return out  # [B, 3, 256, 256] in [0,1] RGB


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--out", default=os.path.join(UPSTREAM, "models/sd-vae/vae_decoder_b{batch}.mlpackage"))
    ap.add_argument("--compute-units", choices=["all", "cpu_and_gpu", "cpu_and_ne"], default="all")
    args = ap.parse_args()
    out_path = args.out.format(batch=args.batch)

    print(f"[coreml] loading VAE from {VAE_DIR}")
    vae = AutoencoderKL.from_pretrained(VAE_DIR)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    scaling = float(vae.config.scaling_factor)

    model = VAEDecoder(vae, scaling).eval()
    B = args.batch
    example = torch.randn(B, 4, 32, 32)

    print(f"[coreml] tracing (batch={B})…")
    t0 = time.time()
    with torch.no_grad():
        traced = torch.jit.trace(model, example, strict=False)
    print(f"[coreml] traced in {time.time()-t0:.1f}s")

    cu = {
        "all": ct.ComputeUnit.ALL,
        "cpu_and_gpu": ct.ComputeUnit.CPU_AND_GPU,
        "cpu_and_ne": ct.ComputeUnit.CPU_AND_NE,
    }[args.compute_units]

    print(f"[coreml] converting → fp16, compute_units={args.compute_units}")
    t0 = time.time()
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="latents", shape=example.shape, dtype=float)],
        outputs=[ct.TensorType(name="image", dtype=float)],
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        compute_units=cu,
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
