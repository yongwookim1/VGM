"""Dataset for SafeGem video safety training."""

from __future__ import annotations

import logging
import os

import torch
from torch.utils.data import Dataset

from src.common.io import load_json
from src.data.video_utils import sample_frames_from_video

logger = logging.getLogger(__name__)


class VideoSafetyDataset(Dataset):
    """Dataset for video safety training with SafeGem-12B."""

    def __init__(
        self,
        data_path: str,
        processor,
        max_frames: int = 8,
        fps: float = 1.0,
        max_length: int = 8192,
        skip_missing_videos: bool = True,
    ):
        self.data = load_json(data_path)
        self.processor = processor
        self.max_frames = max_frames
        self.fps = fps
        self.max_length = max_length
        self.ignore_index = -100

        if skip_missing_videos:
            original_len = len(self.data)
            self.data = [sample for sample in self.data if os.path.isfile(sample["video_path"])]
            if len(self.data) < original_len:
                logger.warning(
                    "Filtered %s samples with missing videos (%s remaining)",
                    original_len - len(self.data),
                    len(self.data),
                )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        try:
            frames = sample_frames_from_video(sample["video_path"], self.max_frames, self.fps)
        except Exception as exc:
            logger.warning("Error loading video %s: %s", sample["video_path"], exc)
            return None

        user_content = [{"type": "image"} for _ in frames]
        user_content.append({"type": "text", "text": sample["question"]})
        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": [{"type": "text", "text": sample["answer"]}]},
        ]
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        inputs = self.processor(
            text=[text],
            images=frames,
            padding=False,
            return_tensors="pt",
        )

        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)
        if input_ids.shape[0] > self.max_length:
            input_ids = input_ids[: self.max_length]
            attention_mask = attention_mask[: self.max_length]

        labels = self._mask_non_assistant_tokens(input_ids.clone())
        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "safety_labels": torch.tensor(sample["safety_label"], dtype=torch.long),
            "num_frames": torch.tensor(len(frames), dtype=torch.long),
        }
        pixel_values = inputs.get("pixel_values")
        if pixel_values is not None:
            # Gemma processors may return [1, T, C, H, W]; keep per-frame batches as 4D.
            result["pixel_values"] = (
                pixel_values.squeeze(0)
                if pixel_values.dim() == 5 and pixel_values.shape[0] == 1
                else pixel_values
            )
        return result

    def _mask_non_assistant_tokens(self, labels: torch.Tensor) -> torch.Tensor:
        tokenizer = self.processor.tokenizer
        start_of_turn_id = tokenizer.convert_tokens_to_ids("<start_of_turn>")
        start_positions = (labels == start_of_turn_id).nonzero(as_tuple=True)[0]
        if len(start_positions) == 0:
            return labels

        last_start = start_positions[-1].item()
        response_start = last_start + 1
        newline_id = tokenizer.convert_tokens_to_ids("\n")
        for i in range(last_start + 1, min(last_start + 5, len(labels))):
            if labels[i].item() == newline_id:
                response_start = i + 1
                break

        labels[:response_start] = self.ignore_index
        return labels
