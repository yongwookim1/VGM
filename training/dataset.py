"""
VideoSafetyDataset: loads video QA samples and formats them for
SafeLLaVA-7B training with binary safety labels.

Since LLaVA handles single images, we sample one representative frame
per video (or optionally multiple frames concatenated).
"""

import json
import logging
import os

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


def sample_frames_from_video(video_path: str, max_frames: int = 8,
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

    num_frames_by_fps = max(1, int(duration * fps))
    num_frames = min(num_frames_by_fps, max_frames)
    num_frames = max(1, num_frames)

    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(idx) - 1))
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame_rgb))

    cap.release()

    if len(frames) == 0:
        raise ValueError(f"Could not read any frames from: {video_path}")

    return frames


def select_representative_frame(frames: list[Image.Image]) -> Image.Image:
    """Select the middle frame as the representative frame."""
    return frames[len(frames) // 2]


class VideoSafetyDataset(Dataset):
    """Dataset for video safety training with SafeLLaVA-7B.

    SafeLLaVA processes single images, so we extract a representative
    frame from each video. The frame is preprocessed using CLIP's
    image processor via safellava utilities.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer,
        image_processor,
        model_config,
        max_frames: int = 8,
        fps: float = 1.0,
        max_length: int = 2048,
        skip_missing_videos: bool = True,
    ):
        with open(data_path) as f:
            self.data = json.load(f)

        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config
        self.max_frames = max_frames
        self.fps = fps
        self.max_length = max_length
        self.skip_missing_videos = skip_missing_videos

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

        # Import SafeLLaVA utilities
        from safellava.mm_utils import process_images, tokenizer_image_token
        from safellava.conversation import conv_templates
        from safellava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN

        self._process_images = process_images
        self._tokenizer_image_token = tokenizer_image_token
        self._conv_templates = conv_templates
        self._IMAGE_TOKEN_INDEX = IMAGE_TOKEN_INDEX
        self._DEFAULT_IMAGE_TOKEN = DEFAULT_IMAGE_TOKEN
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

        # Select representative frame for LLaVA (single image model)
        image = select_representative_frame(frames)

        # Process image through CLIP processor
        image_tensor = self._process_images(
            [image], self.image_processor, self.model_config
        )
        if image_tensor.dim() == 4:
            image_tensor = image_tensor[0]  # [C, H, W]

        # Build conversation using LLaVA v1 template
        conv = self._conv_templates["llava_v1"].copy()
        question = self._DEFAULT_IMAGE_TOKEN + "\n" + sample["question"]
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], sample["answer"])
        prompt = conv.get_prompt()

        # Tokenize with image token handling
        input_ids = self._tokenizer_image_token(
            prompt, self.tokenizer, self._IMAGE_TOKEN_INDEX, return_tensors="pt"
        )

        # Truncate if needed
        if input_ids.shape[0] > self.max_length:
            input_ids = input_ids[:self.max_length]

        # Build labels: mask everything except assistant response
        labels = input_ids.clone()
        labels = self._mask_non_assistant_tokens(labels, prompt)

        safety_label = torch.tensor(sample["safety_label"], dtype=torch.long)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "image": image_tensor,
            "image_size": image.size,  # (width, height)
            "safety_labels": safety_label,
        }

    def _mask_non_assistant_tokens(self, labels: torch.Tensor, prompt: str) -> torch.Tensor:
        """Mask all tokens except the assistant's response.

        LLaVA v1 format:
        <system> USER: <image>\n<question> ASSISTANT: <answer></s>

        We mask everything up to and including "ASSISTANT: ".
        """
        # Find "ASSISTANT:" in the prompt and compute its token position
        assistant_marker = "ASSISTANT:"
        marker_pos = prompt.rfind(assistant_marker)
        if marker_pos == -1:
            return labels

        # Tokenize up to the marker to find the split point
        prefix = prompt[:marker_pos + len(assistant_marker)]
        # Use the image token-aware tokenizer
        prefix_ids = self._tokenizer_image_token(
            prefix, self.tokenizer, self._IMAGE_TOKEN_INDEX, return_tensors="pt"
        )
        split_point = prefix_ids.shape[0]

        # Mask everything before and including the marker
        labels[:split_point] = self.ignore_index

        return labels
