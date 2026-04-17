"""Compare CoreML UNet latency across compute unit choices vs PyTorch MPS."""
import os
import sys
import time

import coremltools as ct
import numpy as np
import torch
from diffusers import UNet2DConditionModel

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPSTREAM = os.path.join(ROOT, "upstream")
UNET_CFG = os.path.join(UPSTREAM, "models/musetalkV15/musetalk.json")
UNET_W = os.path.join(UPSTREAM, "models/musetalkV15/unet.pth")
MLPKG = os.path.join(UPSTREAM, "models/musetalkV15/unet_b16.mlpackage")

B = 16
WARMUP = 2
RUNS = 5


def bench_coreml(compute_unit_name):
    cu = {
        "ALL": ct.ComputeUnit.ALL,
        "CPU_AND_GPU": ct.ComputeUnit.CPU_AND_GPU,
        "CPU_AND_NE": ct.ComputeUnit.CPU_AND_NE,
        "CPU_ONLY": ct.ComputeUnit.CPU_ONLY,
    }[compute_unit_name]
    print(f"[coreml:{compute_unit_name}] loading {MLPKG}")
    t0 = time.time()
    m = ct.models.MLModel(MLPKG, compute_units=cu)
    load_s = time.time() - t0

    rng = np.random.default_rng(0)
    sample = rng.standard_normal((B, 8, 32, 32)).astype(np.float32)
    timesteps = np.zeros((B,), dtype=np.int32)
    ehs = rng.standard_normal((B, 50, 384)).astype(np.float32)

    for _ in range(WARMUP):
        _ = m.predict({"sample": sample, "timesteps": timesteps, "encoder_hidden_states": ehs})

    dur = []
    for _ in range(RUNS):
        t0 = time.time()
        out = m.predict({"sample": sample, "timesteps": timesteps, "encoder_hidden_states": ehs})
        dur.append(time.time() - t0)
    dur = np.array(dur) * 1000
    print(f"[coreml:{compute_unit_name}] load={load_s:.1f}s  mean={dur.mean():.0f}ms  min={dur.min():.0f}ms  max={dur.max():.0f}ms  (B={B})")
    return out["predicted_sample"]


def bench_pytorch_mps():
    device = torch.device("mps")
    print(f"[pytorch:mps] loading UNet")
    unet = UNet2DConditionModel.from_config(UNET_CFG)
    state = torch.load(UNET_W, map_location="cpu", weights_only=False)
    unet.load_state_dict(state)
    unet.to(device).eval()

    sample = torch.randn(B, 8, 32, 32, device=device)
    timesteps = torch.zeros(B, dtype=torch.int32, device=device)
    ehs = torch.randn(B, 50, 384, device=device)

    with torch.no_grad():
        for _ in range(WARMUP):
            _ = unet(sample, timesteps, encoder_hidden_states=ehs).sample
        torch.mps.synchronize()

        dur = []
        for _ in range(RUNS):
            t0 = time.time()
            out = unet(sample, timesteps, encoder_hidden_states=ehs).sample
            torch.mps.synchronize()
            dur.append(time.time() - t0)
    dur = np.array(dur) * 1000
    print(f"[pytorch:mps] mean={dur.mean():.0f}ms  min={dur.min():.0f}ms  max={dur.max():.0f}ms  (B={B})")
    return out.detach().cpu().numpy()


if __name__ == "__main__":
    ref = bench_pytorch_mps()
    for cu in ["ALL", "CPU_AND_NE", "CPU_AND_GPU", "CPU_ONLY"]:
        try:
            out = bench_coreml(cu)
            # Quick numerical parity check (rough — CoreML fp16)
            diff = np.abs(out.astype(np.float32) - ref).mean()
            rel = diff / (np.abs(ref).mean() + 1e-6)
            print(f"  parity: mean_abs_diff={diff:.4f}  rel_err={rel:.3%}")
        except Exception as e:
            print(f"[coreml:{cu}] FAILED: {e}")
