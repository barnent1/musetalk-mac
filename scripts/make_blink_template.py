"""Build a synthetic LivePortrait motion template (.pkl) that holds identity
motion but injects clean full-closure blinks into c_eyes_lst.

We copy the per-frame motion dict from a reference template so LivePortrait
has valid motion params, then overwrite eye/lip scalars with our signal.
"""
import argparse
import copy
import pickle
import random
from pathlib import Path

import numpy as np


# Eye-closure scalar ranges observed from real Kling driver templates:
#   fully open  ~ 0.40  (range 0.30–0.45)
#   fully closed ~ 0.02  (range 0.01–0.05)
OPEN_RATIO = 0.40
CLOSED_RATIO = 0.02

# A natural blink: ramp closed over ~2 frames, hold ~1, ramp open over ~2
BLINK_PROFILE = [
    OPEN_RATIO * 0.55,         # mid-close
    CLOSED_RATIO + 0.02,       # near-closed
    CLOSED_RATIO,              # fully closed
    CLOSED_RATIO + 0.04,       # starting to open
    OPEN_RATIO * 0.70,         # mostly open
]


def build_blink_signal(n_frames: int, fps: float, every_sec: float = 3.5,
                       jitter: float = 0.8, seed: int = 42) -> np.ndarray:
    """Return an (N, 1, 2) float32 array with crisp blinks every ~every_sec."""
    rng = random.Random(seed)
    signal = np.full((n_frames, 1, 2), OPEN_RATIO, dtype=np.float32)

    # First blink slightly offset so the video doesn't open on a blink
    t = every_sec * 0.6
    while t < n_frames / fps:
        start_frame = int(t * fps)
        for i, val in enumerate(BLINK_PROFILE):
            idx = start_frame + i
            if 0 <= idx < n_frames:
                signal[idx, 0, 0] = val
                signal[idx, 0, 1] = val
        t += every_sec + rng.uniform(-jitter, jitter)
    return signal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="liveportrait/assets/examples/driving/d0.pkl",
                    help="reference template to borrow motion-frame dict structure from")
    ap.add_argument("--out", required=True, help="output .pkl path")
    ap.add_argument("--frames", type=int, required=True, help="number of frames")
    ap.add_argument("--fps", type=float, required=True)
    ap.add_argument("--every", type=float, default=3.5,
                    help="average seconds between blinks")
    args = ap.parse_args()

    with open(args.ref, "rb") as f:
        ref = pickle.load(f)

    # Use frame 0 of the reference for every frame → "identity" driver motion.
    # (Combined with --animation-region lip at inference time, the motion is
    # ignored anyway; we only need a valid structure.)
    ref_motion = copy.deepcopy(ref["motion"][0])
    motion = [copy.deepcopy(ref_motion) for _ in range(args.frames)]

    c_eyes = build_blink_signal(args.frames, args.fps, every_sec=args.every)
    c_eyes_lst = [c_eyes[i] for i in range(args.frames)]

    # Static lips — no lip motion transfer
    c_lip_default = np.array([[0.0]], dtype=np.float32)
    c_lip_lst = [c_lip_default.copy() for _ in range(args.frames)]

    out = {
        "n_frames": args.frames,
        "output_fps": int(round(args.fps)),
        "motion": motion,
        "c_eyes_lst": c_eyes_lst,
        "c_lip_lst": c_lip_lst,
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(out, f)

    blink_frames = [i for i, v in enumerate(c_eyes_lst) if v[0, 0] < 0.1]
    print(f"wrote {args.out}: {args.frames} frames @ {args.fps} fps, "
          f"{len(blink_frames)} near-closed frames across {len(blink_frames)//3} blinks")


if __name__ == "__main__":
    main()
