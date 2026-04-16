"""Dataset utilities for SafeQwen2.5-VL video safety training."""

from __future__ import annotations

import json
import logging
import math
import os

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

_PATCH_FACTOR = 28
_MAX_VIDEO_PIXELS = 256 * _PATCH_FACTOR * _PATCH_FACTOR

SYSTEM_PROMPT = (
    "You are a helpful and safe AI assistant. When presented with video content, "
    "analyze it carefully. If the content or request involves harmful, unsafe, or "
    "inappropriate material, politely decline and explain why. Otherwise, provide "
    "a helpful and accurate response."
)


def _resize_frame(img: Image.Image, max_pixels: int = _MAX_VIDEO_PIXELS) -> Image.Image:
    """Resize a frame to preserve Qwen2.5-VL token alignment."""
    width, height = img.size
    if width * height > max_pixels:
        scale = math.sqrt(max_pixels / (width * height))
        width = max(_PATCH_FACTOR, int(width * scale / _PATCH_FACTOR) * _PATCH_FACTOR)
        height = max(_PATCH_FACTOR, int(height * scale / _PATCH_FACTOR) * _PATCH_FACTOR)
    else:
        width = max(_PATCH_FACTOR, round(width / _PATCH_FACTOR) * _PATCH_FACTOR)
        height = max(_PATCH_FACTOR, round(height / _PATCH_FACTOR) * _PATCH_FACTOR)
    return img.resize((width, height), Image.BILINEAR)


def sample_frames_from_video(video_path: str, max_frames: int = 16, fps: float = 1.0) -> list[Image.Image]:
    """Uniformly sample an even number of frames from a video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0:
        video_fps = 30.0
    duration = total_frames / video_fps if total_frames > 0 else 0

    num_frames_by_fps = max(1, int(duration * fps))
    num_frames = min(num_frames_by_fps, max_frames)
    num_frames = max(2, num_frames - (num_frames % 2))

    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(idx) - 1))
            ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))

    cap.release()

    if not frames:
        raise ValueError(f"Could not read any frames from: {video_path}")
    if len(frames) % 2 != 0:
        frames = frames[:-1]
    if not frames:
        raise ValueError(f"Not enough frames after alignment: {video_path}")
    return frames


class VideoSafetyDataset(Dataset):
    """Dataset for SafeQwen2.5-VL video safety fine-tuning."""

    def __init__(
        self,
        data_path: str,
        processor,
        max_frames: int = 16,
        fps: float = 1.0,
        max_length: int = 2048,
        skip_missing_videos: bool = True,
    ):
        with open(data_path) as f:
            self.data = json.load(f)

        self.processor = processor
        self.max_frames = max_frames
        self.fps = fps
        self.max_length = max_length
        self.skip_missing_videos = skip_missing_videos

        if skip_missing_videos:
            original_len = len(self.data)
            self.data = [sample for sample in self.data if os.path.isfile(sample["video_path"])]
            if len(self.data) < original_len:
                logger.warning(
                    "Filtered %s samples with missing videos (%s remaining)",
                    original_len - len(self.data),
                    len(self.data),
                )

        self.im_start_id = processor.tokenizer.convert_tokens_to_ids("<|im_start|>")
        self.video_pad_id = processor.tokenizer.convert_tokens_to_ids("<|video_pad|>")
        self.ignore_index = -100

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        try:
            frames = sample_frames_from_video(sample["video_path"], self.max_frames, self.fps)
        except Exception as exc:
            logger.warning("Error loading video %s: %s", sample["video_path"], exc)
            return None

        frames = [_resize_frame(frame) for frame in frames]
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": frames},
                    {"type": "text", "text": sample["question"]},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": sample["answer"]}],
            },
        ]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        inputs = self.processor(
            text=[text],
            videos=[frames],
            padding=False,
            return_tensors="pt",
        )

        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)

        if input_ids.shape[0] > self.max_length:
            truncated_ids = input_ids[: self.max_length]
            full_video_tokens = (input_ids == self.video_pad_id).sum().item()
            truncated_video_tokens = (truncated_ids == self.video_pad_id).sum().item()
            if truncated_video_tokens != full_video_tokens:
                logger.warning(
                    "Skipping sample %s: truncation to %s would remove %s video placeholder tokens (seq_len=%s)",
                    idx,
                    self.max_length,
                    full_video_tokens - truncated_video_tokens,
                    input_ids.shape[0],
                )
                return None
            input_ids = truncated_ids
            attention_mask = attention_mask[: self.max_length]

        labels = self._mask_non_assistant_tokens(input_ids.clone())
        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "safety_labels": torch.tensor(sample["safety_label"], dtype=torch.long),
        }

        pixel_values_videos = inputs.get("pixel_values_videos")
        video_grid_thw = inputs.get("video_grid_thw")
        if pixel_values_videos is not None:
            result["pixel_values_videos"] = pixel_values_videos.squeeze(0) if pixel_values_videos.dim() == 3 else pixel_values_videos
        if video_grid_thw is not None:
            result["video_grid_thw"] = video_grid_thw
        return result

    def _mask_non_assistant_tokens(self, labels: torch.Tensor) -> torch.Tensor:
        im_start_positions = (labels == self.im_start_id).nonzero(as_tuple=True)[0]
        if len(im_start_positions) == 0:
            return labels

        last_im_start = im_start_positions[-1].item()
        labels[:last_im_start] = self.ignore_index
        assistant_window = labels[last_im_start:]
        newline_id = self.processor.tokenizer.encode("\n", add_special_tokens=False)
        start_offset = 1
        if newline_id:
            newline_positions = (assistant_window == newline_id[-1]).nonzero(as_tuple=True)[0]
            if len(newline_positions) > 0:
                start_offset = newline_positions[0].item() + 1
        labels[last_im_start : last_im_start + start_offset] = self.ignore_index
        return labels
