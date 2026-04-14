"""Benchmark TAESD decoder vs SD VAE + CoreML VAE at batch 16."""
import os
import time

import coremltools as ct
import numpy as np
import torch
from diffusers import AutoencoderKL, AutoencoderTiny

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPSTREAM = os.path.join(ROOT, "upstream")
SD_VAE_DIR = os.path.join(UPSTREAM, "models/sd-vae")
TAESD_DIR = os.path.join(UPSTREAM, "models/taesd")
COREML_VAE = os.path.join(UPSTREAM, "models/sd-vae/vae_decoder_b16.mlpackage")

B = 16
WARMUP = 2
RUNS = 5

np.random.seed(42)
latents_np = np.random.standard_normal((B, 4, 32, 32)).astype(np.float32) * 3.0


def bench_sd_vae_mps():
    device = torch.device("mps")
    vae = AutoencoderKL.from_pretrained(SD_VAE_DIR).to(device).eval()
    scaling = float(vae.config.scaling_factor)
    latents = torch.from_numpy(latents_np).to(device)
    with torch.no_grad():
        for _ in range(WARMUP):
            out = vae.decode(latents / scaling).sample
            torch.mps.synchronize()
        dur = []
        for _ in range(RUNS):
            t0 = time.time()
            out = vae.decode(latents / scaling).sample
            out = (out / 2 + 0.5).clamp(0, 1)
            torch.mps.synchronize()
            dur.append(time.time() - t0)
    d = np.array(dur) * 1000
    print(f"[sd_vae:mps   ] mean={d.mean():.0f}ms min={d.min():.0f}ms")
    return out.float().cpu().numpy()


def bench_taesd_mps():
    device = torch.device("mps")
    taesd = AutoencoderTiny.from_pretrained(TAESD_DIR).to(device).eval()
    # TAESD takes unscaled SD latents (what UNet outputs directly, no scaling divide)
    latents = torch.from_numpy(latents_np).to(device)
    with torch.no_grad():
        for _ in range(WARMUP):
            out = taesd.decode(latents).sample
            torch.mps.synchronize()
        dur = []
        for _ in range(RUNS):
            t0 = time.time()
            out = taesd.decode(latents).sample
            # TAESD outputs in [0, 1] already
            torch.mps.synchronize()
            dur.append(time.time() - t0)
    d = np.array(dur) * 1000
    print(f"[taesd:mps    ] mean={d.mean():.0f}ms min={d.min():.0f}ms")
    return out.float().cpu().numpy()


def bench_coreml_sd(cu_name):
    cu = {
        "ALL": ct.ComputeUnit.ALL,
        "CPU_AND_GPU": ct.ComputeUnit.CPU_AND_GPU,
    }[cu_name]
    m = ct.models.MLModel(COREML_VAE, compute_units=cu)
    for _ in range(WARMUP):
        _ = m.predict({"latents": latents_np})
    dur = []
    for _ in range(RUNS):
        t0 = time.time()
        out = m.predict({"latents": latents_np})
        dur.append(time.time() - t0)
    d = np.array(dur) * 1000
    print(f"[coreml_sd:{cu_name}] mean={d.mean():.0f}ms min={d.min():.0f}ms")
    return out["image"]


if __name__ == "__main__":
    ref_sd = bench_sd_vae_mps()
    taesd_out = bench_taesd_mps()
    cm_sd = bench_coreml_sd("CPU_AND_GPU")

    print()
    diff_taesd = np.abs(taesd_out.astype(np.float32) - ref_sd).mean()
    rel_taesd = diff_taesd / (np.abs(ref_sd).mean() + 1e-6)
    print(f"TAESD vs SD VAE parity: mean_abs_diff={diff_taesd:.4f}  rel_err={rel_taesd:.2%}")

    diff_cm = np.abs(cm_sd.astype(np.float32) - ref_sd).mean()
    rel_cm = diff_cm / (np.abs(ref_sd).mean() + 1e-6)
    print(f"CoreML-SD vs SD VAE parity: mean_abs_diff={diff_cm:.4f}  rel_err={rel_cm:.2%}")
