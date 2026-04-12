"""
VideoSafetyDataset: loads video QA samples and formats them for
SafeQwen2.5-VL training with safety labels.
"""

import json
import logging
import math
import os
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


def sample_frames_from_video(video_path: str, max_frames: int = 16,
                             fps: float = 1.0) -> list[Image.Image]:
    """Uniformly sample frames from a video file using OpenCV."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0:
        video_fps = 30.0
    duration = total_frames / video_fps

    # Calculate how many frames to sample
    num_frames_by_fps = max(1, int(duration * fps))
    num_frames = min(num_frames_by_fps, max_frames)
    # Must be even (temporal_patch_size=2 in Qwen2.5-VL)
    num_frames = max(2, num_frames - (num_frames % 2))

    # Uniform sampling indices
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
        else:
            # Retry with nearest frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(idx) - 1))
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame_rgb))

    cap.release()

    if len(frames) == 0:
        raise ValueError(f"Could not read any frames from: {video_path}")

    # Ensure even number of frames
    if len(frames) % 2 != 0:
        frames = frames[:-1]
    if len(frames) == 0:
        raise ValueError(f"Not enough frames after alignment: {video_path}")

    return frames


SYSTEM_PROMPT = (
    "You are a helpful and safe AI assistant. When presented with video content, "
    "analyze it carefully. If the content or request involves harmful, unsafe, or "
    "inappropriate material, politely decline and explain why. Otherwise, provide "
    "a helpful and accurate response."
)


class VideoSafetyDataset(Dataset):
    """Dataset for video safety training with SafeQwen2.5-VL.

    Each sample contains:
    - Video frames formatted for Qwen2.5-VL processor
    - Question-answer pair as chat messages
    - Safety label for the safety classification head
    """

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

        # Filter out samples with missing videos if requested
        if skip_missing_videos:
            original_len = len(self.data)
            self.data = [
                s for s in self.data if os.path.isfile(s["video_path"])
            ]
            if len(self.data) < original_len:
                logger.warning(
                    f"Filtered {original_len - len(self.data)} samples with "
                    f"missing videos ({len(self.data)} remaining)"
                )

        # Get special token IDs for label masking
        self.im_start_id = processor.tokenizer.convert_tokens_to_ids("<|im_start|>")
        self.im_end_id = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.ignore_index = -100

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        try:
            frames = sample_frames_from_video(
                sample["video_path"], self.max_frames, self.fps
            )
        except Exception as e:
            logger.warning(f"Error loading video {sample['video_path']}: {e}")
            return None

        # Build chat messages
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

        # Apply chat template to get full text with special tokens
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        # Process with the Qwen2.5-VL processor
        # NOTE: do NOT use truncation=True here — truncating at processor level
        # breaks the video token count consistency check between text and input_ids.
        # Instead, truncate the resulting tensors manually after processing.
        inputs = self.processor(
            text=[text],
            videos=[frames],
            padding=False,
            return_tensors="pt",
        )

        # Squeeze batch dimension (processor returns [1, seq_len])
        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)

        # Manual truncation after processing (safe — tensors only, no token mismatch)
        if input_ids.shape[0] > self.max_length:
            input_ids = input_ids[:self.max_length]
            attention_mask = attention_mask[:self.max_length]
        pixel_values_videos = inputs.get("pixel_values_videos")
        video_grid_thw = inputs.get("video_grid_thw")

        # Build labels: mask everything except the assistant's response
        labels = input_ids.clone()
        labels = self._mask_non_assistant_tokens(labels)

        safety_label = torch.tensor(sample["safety_label"], dtype=torch.long)

        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "safety_labels": safety_label,
        }
        if pixel_values_videos is not None:
            result["pixel_values_videos"] = pixel_values_videos.squeeze(0) \
                if pixel_values_videos.dim() > 2 else pixel_values_videos
        if video_grid_thw is not None:
            result["video_grid_thw"] = video_grid_thw

        return result

    def _mask_non_assistant_tokens(self, labels: torch.Tensor) -> torch.Tensor:
        """Mask all tokens except the assistant's response with -100.

        Chat format: <|im_start|>system\n...<|im_end|>\n
                     <|im_start|>user\n...<|im_end|>\n
                     <|im_start|>assistant\n RESPONSE <|im_end|>

        We find the last <|im_start|> and unmask from the token after
        "assistant\n" up to and including the final <|im_end|>.
        """
        im_start_positions = (labels == self.im_start_id).nonzero(as_tuple=True)[0]

        if len(im_start_positions) == 0:
            return labels

        # The last <|im_start|> is the assistant turn
        last_im_start = im_start_positions[-1].item()

        # Mask everything before the assistant response content.
        # The format after <|im_start|> is: "assistant\n" (several tokens)
        # We find the response start by looking for the \n after "assistant"
        # A simpler approach: mask up to last_im_start + offset
        # "assistant\n" is typically 2 tokens: "assistant" + "\n"
        # But let's find it precisely by scanning for the first non-special token
        # after the assistant role marker

        # Find the first \n token after last_im_start (role designation ends at \n)
        response_start = last_im_start + 1
        # Skip "assistant" and "\n" tokens (usually positions +1 and +2)
        # Scan forward to find the newline after role name
        newline_id = self.processor.tokenizer.convert_tokens_to_ids("\n")
        for i in range(last_im_start + 1, min(last_im_start + 5, len(labels))):
            if labels[i].item() == newline_id:
                response_start = i + 1
                break

        # Mask everything before response_start
        labels[:response_start] = self.ignore_index

        return labels
