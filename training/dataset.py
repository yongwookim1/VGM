"""
VideoSafetyDataset for binary safety training.

Two dataset classes:
- VideoSafetyDatasetSafeGem: SafeGem-12B (Gemma3). Frames as multiple images.
- VideoSafetyDatasetLLaVA: SafeLLaVA-7B (LLaMA+CLIP). Single representative frame.
"""

import json
import logging
import os
from typing import Optional

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


class VideoSafetyDatasetSafeGem(Dataset):
    """Dataset for video safety training with SafeGem-12B.

    Each sample contains:
    - Video frames passed as multiple images to Gemma3 processor
    - Question-answer pair as chat messages
    - Binary safety label for the safety classification head
    """

    def __init__(
        self,
        data_path: str,
        processor,
        max_frames: int = 8,
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
            self.data = [
                s for s in self.data if os.path.isfile(s["video_path"])
            ]
            if len(self.data) < original_len:
                logger.warning(
                    f"Filtered {original_len - len(self.data)} samples with "
                    f"missing videos ({len(self.data)} remaining)"
                )

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

        # Build chat messages: each frame as a separate image in the user content
        user_content = []
        for _frame in frames:
            user_content.append({"type": "image"})
        user_content.append({"type": "text", "text": sample["question"]})

        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": [{"type": "text", "text": sample["answer"]}]},
        ]

        # Apply chat template
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        # Process with Gemma3 processor
        inputs = self.processor(
            text=[text],
            images=frames,
            padding=False,
            return_tensors="pt",
        )

        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)

        # Truncate if needed
        if input_ids.shape[0] > self.max_length:
            input_ids = input_ids[:self.max_length]
            attention_mask = attention_mask[:self.max_length]

        # Build labels: mask everything except the assistant's response
        labels = input_ids.clone()
        labels = self._mask_non_assistant_tokens(labels)

        safety_label = torch.tensor(sample["safety_label"], dtype=torch.long)

        # Get pixel values from processor output
        pixel_values = inputs.get("pixel_values")

        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "safety_labels": safety_label,
        }
        if pixel_values is not None:
            result["pixel_values"] = pixel_values.squeeze(0) \
                if pixel_values.dim() > 3 and pixel_values.shape[0] == 1 else pixel_values

        return result

    def _mask_non_assistant_tokens(self, labels: torch.Tensor) -> torch.Tensor:
        """Mask all tokens except the assistant's response with -100.

        Gemma3 chat format uses <start_of_turn> / <end_of_turn> markers.
        We find the last assistant turn and unmask only its content.
        """
        tokenizer = self.processor.tokenizer

        start_of_turn_id = tokenizer.convert_tokens_to_ids("<start_of_turn>")
        end_of_turn_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")

        # Find all <start_of_turn> positions
        start_positions = (labels == start_of_turn_id).nonzero(as_tuple=True)[0]

        if len(start_positions) == 0:
            return labels

        # The last <start_of_turn> is the assistant turn
        last_start = start_positions[-1].item()

        # Find the response start: after "<start_of_turn>model\n"
        response_start = last_start + 1
        newline_id = tokenizer.convert_tokens_to_ids("\n")
        for i in range(last_start + 1, min(last_start + 5, len(labels))):
            if labels[i].item() == newline_id:
                response_start = i + 1
                break

        # Mask everything before response_start
        labels[:response_start] = self.ignore_index

        return labels


# ---------------------------------------------------------------------------
# SafeLLaVA-7B dataset
# ---------------------------------------------------------------------------
class VideoSafetyDatasetLLaVA(Dataset):
    """Dataset for video safety training with SafeLLaVA-7B.

    LLaVA processes single images, so we pick the middle frame.
    Uses CLIP image processor and LLaVA v1 conversation template.
    """

    def __init__(self, data_path, tokenizer, image_processor, model_config,
                 max_frames=8, fps=1.0, max_length=2048, skip_missing_videos=True):
        with open(data_path) as f:
            self.data = json.load(f)

        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config
        self.max_frames = max_frames
        self.fps = fps
        self.max_length = max_length
        self.ignore_index = -100

        if skip_missing_videos:
            original_len = len(self.data)
            self.data = [s for s in self.data if os.path.isfile(s["video_path"])]
            if len(self.data) < original_len:
                logger.warning(
                    f"Filtered {original_len - len(self.data)} missing videos "
                    f"({len(self.data)} remaining)"
                )

        from safellava.mm_utils import process_images, tokenizer_image_token
        from safellava.conversation import conv_templates
        from safellava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
        self._process_images = process_images
        self._tokenizer_image_token = tokenizer_image_token
        self._conv_templates = conv_templates
        self._IMAGE_TOKEN_INDEX = IMAGE_TOKEN_INDEX
        self._DEFAULT_IMAGE_TOKEN = DEFAULT_IMAGE_TOKEN

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        try:
            frames = sample_frames_from_video(sample["video_path"], self.max_frames, self.fps)
        except Exception as e:
            logger.warning(f"Error loading video {sample['video_path']}: {e}")
            return None

        # Pick middle frame
        image = frames[len(frames) // 2]

        # Process through CLIP
        image_tensor = self._process_images([image], self.image_processor, self.model_config)
        if image_tensor.dim() == 4:
            image_tensor = image_tensor[0]

        # Build LLaVA v1 conversation
        conv = self._conv_templates["llava_v1"].copy()
        question = self._DEFAULT_IMAGE_TOKEN + "\n" + sample["question"]
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], sample["answer"])
        prompt = conv.get_prompt()

        input_ids = self._tokenizer_image_token(
            prompt, self.tokenizer, self._IMAGE_TOKEN_INDEX, return_tensors="pt"
        )

        if input_ids.shape[0] > self.max_length:
            input_ids = input_ids[:self.max_length]

        labels = input_ids.clone()
        labels = self._mask_non_assistant_tokens(labels, prompt)
        safety_label = torch.tensor(sample["safety_label"], dtype=torch.long)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "image": image_tensor,
            "image_size": image.size,
            "safety_labels": safety_label,
        }

    def _mask_non_assistant_tokens(self, labels, prompt):
        """Mask everything before and including 'ASSISTANT: '."""
        marker = "ASSISTANT:"
        pos = prompt.rfind(marker)
        if pos == -1:
            return labels
        prefix = prompt[:pos + len(marker)]
        prefix_ids = self._tokenizer_image_token(
            prefix, self.tokenizer, self._IMAGE_TOKEN_INDEX, return_tensors="pt"
        )
        labels[:prefix_ids.shape[0]] = self.ignore_index
        return labels
