"""Shared video sampling and frame resizing utilities."""

from __future__ import annotations

import numpy as np
from PIL import Image


def sample_frames_from_video(
    video_path: str,
    max_frames: int = 8,
    fps: float = 1.0,
) -> list[Image.Image]:
    """Uniformly sample frames from a video file using OpenCV.

    The sampled frame indices stay unchanged, but frames are decoded in
    ascending order to avoid repeated random seeks on compressed videos.
    """
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0:
        video_fps = 30.0
    duration = total_frames / video_fps if total_frames > 0 else 0

    num_frames_by_fps = max(1, int(duration * fps))
    num_frames = max(1, min(num_frames_by_fps, max_frames))
    if total_frames <= 0:
        cap.release()
        raise ValueError(f"Video has no decodable frames: {video_path}")

    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int).tolist()
    target_indices = sorted(dict.fromkeys(int(idx) for idx in indices))

    frames_by_index: dict[int, Image.Image] = {}
    current_target_pos = 0
    current_frame_idx = target_indices[0]
    cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_idx)

    while current_target_pos < len(target_indices):
        ret, frame = cap.read()
        if not ret:
            break

        target_idx = target_indices[current_target_pos]
        if current_frame_idx == target_idx:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames_by_index[target_idx] = Image.fromarray(frame_rgb)
            current_target_pos += 1
        current_frame_idx += 1

    cap.release()

    frames = [frames_by_index[idx] for idx in indices if idx in frames_by_index]

    if not frames:
        raise ValueError(f"Could not read any frames from: {video_path}")
    return frames


def select_representative_frame(frames: list[Image.Image]) -> Image.Image:
    """Select the middle frame as the representative frame."""
    return frames[len(frames) // 2]


def resize_frame(frame: Image.Image, max_pixels: int = 360 * 420) -> Image.Image:
    """Downscale a frame if it exceeds a target pixel budget."""
    width, height = frame.size
    if width * height <= max_pixels:
        return frame

    scale = (max_pixels / float(width * height)) ** 0.5
    resized = frame.resize(
        (max(1, int(width * scale)), max(1, int(height * scale))),
        Image.Resampling.BICUBIC,
    )
    return resized
