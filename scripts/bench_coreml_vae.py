"""Benchmark CoreML VAE decoder against PyTorch MPS using identical inputs."""
import os
import time

import coremltools as ct
import numpy as np
import torch
from diffusers import AutoencoderKL

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPSTREAM = os.path.join(ROOT, "upstream")
VAE_DIR = os.path.join(UPSTREAM, "models/sd-vae")
MLPKG = os.path.join(UPSTREAM, "models/sd-vae/vae_decoder_b16.mlpackage")

B = 16
WARMUP = 2
RUNS = 5

# Identical inputs for both
np.random.seed(42)
latents_np = np.random.standard_normal((B, 4, 32, 32)).astype(np.float32) * 3.0


def bench_pytorch(device_str, use_fp16):
    device = torch.device(device_str)
    print(f"[pytorch:{device_str} fp16={use_fp16}] loading VAE")
    vae = AutoencoderKL.from_pretrained(VAE_DIR).to(device).eval()
    scaling = float(vae.config.scaling_factor)
    if use_fp16:
        vae = vae.half()

    latents = torch.from_numpy(latents_np).to(device)
    if use_fp16:
        latents = latents.half()

    with torch.no_grad():
        for _ in range(WARMUP):
            x = latents * (1.0 / scaling)
            out = vae.decode(x).sample
            if device.type == "mps":
                torch.mps.synchronize()
        dur = []
        for _ in range(RUNS):
            t0 = time.time()
            x = latents * (1.0 / scaling)
            out = vae.decode(x).sample
            out = (out / 2 + 0.5).clamp(0, 1)
            if device.type == "mps":
                torch.mps.synchronize()
            dur.append(time.time() - t0)
    dur = np.array(dur) * 1000
    print(f"[pytorch:{device_str} fp16={use_fp16}] mean={dur.mean():.0f}ms  min={dur.min():.0f}ms")
    return out.float().cpu().numpy()


def bench_coreml(compute_unit_name):
    cu = {
        "ALL": ct.ComputeUnit.ALL,
        "CPU_AND_GPU": ct.ComputeUnit.CPU_AND_GPU,
        "CPU_AND_NE": ct.ComputeUnit.CPU_AND_NE,
    }[compute_unit_name]
    print(f"[coreml:{compute_unit_name}] loading …")
    t0 = time.time()
    m = ct.models.MLModel(MLPKG, compute_units=cu)
    load_s = time.time() - t0

    for _ in range(WARMUP):
        _ = m.predict({"latents": latents_np})
    dur = []
    for _ in range(RUNS):
        t0 = time.time()
        out = m.predict({"latents": latents_np})
        dur.append(time.time() - t0)
    dur = np.array(dur) * 1000
    print(f"[coreml:{compute_unit_name}] load={load_s:.1f}s  mean={dur.mean():.0f}ms  min={dur.min():.0f}ms")
    return out["image"]


if __name__ == "__main__":
    ref = bench_pytorch("mps", use_fp16=False)
    mps_fp16 = bench_pytorch("mps", use_fp16=True)
    for cu in ["ALL", "CPU_AND_GPU", "CPU_AND_NE"]:
        try:
            out = bench_coreml(cu)
            diff = np.abs(out.astype(np.float32) - ref).mean()
            rel = diff / (np.abs(ref).mean() + 1e-6)
            print(f"  parity vs fp32 MPS: mean_abs_diff={diff:.5f}  rel_err={rel:.3%}")
        except Exception as e:
            print(f"[coreml:{cu}] FAILED: {e}")
    # fp16 MPS parity
    d = np.abs(mps_fp16.astype(np.float32) - ref).mean()
    rel = d / (np.abs(ref).mean() + 1e-6)
    print(f"  fp16-MPS parity: mean_abs_diff={d:.5f}  rel_err={rel:.3%}")
