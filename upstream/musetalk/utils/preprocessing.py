"""Face landmark + bbox extraction using MediaPipe Face Mesh.

Replaces the original mmpose/DWPose + face-alignment pipeline. MediaPipe ships
first-class Apple Silicon wheels and has no PyTorch dependency, so inference
runs on macOS without the OpenMMLab source-build headaches.

The public API (``get_landmark_and_bbox``, ``read_imgs``, ``coord_placeholder``)
matches the upstream version so downstream code is unchanged.
"""

import os
import pickle
import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm

coord_placeholder = (0.0, 0.0, 0.0, 0.0)

# MediaPipe Face Mesh — 468 3D face landmarks. Index 6 sits on the nose bridge
# roughly between the eyes, which mirrors the role of the old 68-point ibug
# landmark[29] used in the original algorithm.
_NOSE_BRIDGE_MID = 6

_mp_face_mesh = mp.solutions.face_mesh
_face_mesh = _mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=False,
    min_detection_confidence=0.5,
)


def read_imgs(img_list):
    frames = []
    print("reading images...")
    for img_path in tqdm(img_list):
        frame = cv2.imread(img_path)
        frames.append(frame)
    return frames


def _landmarks_px(frame):
    """Return (N,2) int32 landmark array in pixel coords, or None."""
    h, w = frame.shape[:2]
    # MediaPipe expects RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = _face_mesh.process(rgb)
    if not result.multi_face_landmarks:
        return None
    lm = result.multi_face_landmarks[0].landmark
    pts = np.array([[p.x * w, p.y * h] for p in lm], dtype=np.float32)
    return pts.astype(np.int32)


def get_bbox_range(img_list, upperbondrange=0):
    """Diagnostic helper used by the Gradio app to report bbox_shift ranges."""
    frames = read_imgs(img_list)
    if upperbondrange != 0:
        print("get key_landmark and face bounding boxes with the bbox_shift:", upperbondrange)
    else:
        print("get key_landmark and face bounding boxes with the default value")
    average_range_minus = []
    average_range_plus = []
    for frame in tqdm(frames):
        pts = _landmarks_px(frame)
        if pts is None:
            continue
        # Approximate the old range_minus / range_plus from neighbouring
        # nose-bridge points. MediaPipe 197 ≈ upper-bridge, 6 ≈ mid, 4 ≈ tip.
        mid_y = int(pts[_NOSE_BRIDGE_MID, 1])
        upper_y = int(pts[197, 1])
        tip_y = int(pts[4, 1])
        average_range_minus.append(tip_y - mid_y)
        average_range_plus.append(mid_y - upper_y)

    if not average_range_minus:
        return f"Total frame:「{len(frames)}」 (no faces detected)"
    text_range = (
        f"Total frame:「{len(frames)}」 Manually adjust range : "
        f"[ -{int(sum(average_range_minus) / len(average_range_minus))}~"
        f"{int(sum(average_range_plus) / len(average_range_plus))} ] , "
        f"the current value: {upperbondrange}"
    )
    return text_range


def get_landmark_and_bbox(img_list, upperbondrange=0):
    """Return (coords_list, frames).

    Each coord is a tuple ``(x1, y1, x2, y2)`` suitable for cropping the face
    region for the UNet. Frames with no detected face produce
    ``coord_placeholder`` and are skipped by the caller.
    """
    frames = read_imgs(img_list)
    if upperbondrange != 0:
        print("get key_landmark and face bounding boxes with the bbox_shift:", upperbondrange)
    else:
        print("get key_landmark and face bounding boxes with the default value")

    coords_list = []
    average_range_minus = []
    average_range_plus = []

    for frame in tqdm(frames):
        pts = _landmarks_px(frame)
        if pts is None:
            coords_list.append(coord_placeholder)
            continue

        h, w = frame.shape[:2]
        half_face_coord = pts[_NOSE_BRIDGE_MID].copy()  # (x, y)
        upper_y = int(pts[197, 1])
        tip_y = int(pts[4, 1])
        range_minus = tip_y - int(half_face_coord[1])
        range_plus = int(half_face_coord[1]) - upper_y
        average_range_minus.append(range_minus)
        average_range_plus.append(range_plus)

        if upperbondrange != 0:
            half_face_coord[1] = upperbondrange + half_face_coord[1]

        y_max = int(np.max(pts[:, 1]))
        half_face_dist = y_max - int(half_face_coord[1])
        upper_bond = max(0, int(half_face_coord[1]) - half_face_dist)

        x1 = int(np.min(pts[:, 0]))
        x2 = int(np.max(pts[:, 0]))
        y1 = int(upper_bond)
        y2 = y_max

        # Clip to frame bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        if y2 - y1 <= 0 or x2 - x1 <= 0:
            coords_list.append(coord_placeholder)
        else:
            coords_list.append((x1, y1, x2, y2))

    print("*" * 90)
    if average_range_minus:
        print(
            f"Total frame:「{len(frames)}」 Manually adjust range : "
            f"[ -{int(sum(average_range_minus) / len(average_range_minus))}~"
            f"{int(sum(average_range_plus) / len(average_range_plus))} ] , "
            f"the current value: {upperbondrange}"
        )
    print("*" * 90)
    return coords_list, frames


if __name__ == "__main__":
    img_list = ["./results/lyria/00000.png"]
    coords_list, full_frames = get_landmark_and_bbox(img_list)
    print(coords_list)
