"""Dataset for SafeLLaVA single-frame safety training."""

from __future__ import annotations

import logging
import os

import torch
from torch.utils.data import Dataset

from src.common.io import load_json
from src.data.video_utils import sample_frames_from_video, select_representative_frame

logger = logging.getLogger(__name__)


class VideoSafetyDataset(Dataset):
    """Dataset for video safety training with SafeLLaVA-7B."""

    def __init__(
        self,
        data_path,
        tokenizer,
        image_processor,
        model_config,
        max_frames=8,
        fps=1.0,
        max_length=2048,
        skip_missing_videos=True,
    ):
        self.data = load_json(data_path)
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config
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

        from safellava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
        from safellava.conversation import conv_templates
        from safellava.mm_utils import process_images, tokenizer_image_token

        self._default_image_token = DEFAULT_IMAGE_TOKEN
        self._image_token_index = IMAGE_TOKEN_INDEX
        self._conv_templates = conv_templates
        self._process_images = process_images
        self._tokenizer_image_token = tokenizer_image_token

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        try:
            frames = sample_frames_from_video(sample["video_path"], self.max_frames, self.fps)
        except Exception as exc:
            logger.warning("Error loading video %s: %s", sample["video_path"], exc)
            return None

        image = select_representative_frame(frames)
        image_tensor = self._process_images([image], self.image_processor, self.model_config)
        if image_tensor.dim() == 4:
            image_tensor = image_tensor[0]

        conv = self._conv_templates["llava_v1"].copy()
        question = self._default_image_token + "\n" + sample["question"]
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], sample["answer"])
        prompt = conv.get_prompt()

        input_ids = self._tokenizer_image_token(
            prompt,
            self.tokenizer,
            self._image_token_index,
            return_tensors="pt",
        ).squeeze(0)

        if input_ids.shape[0] > self.max_length:
            input_ids = input_ids[: self.max_length]

        labels = self._mask_non_assistant_tokens(input_ids.clone(), prompt)
        return {
            "input_ids": input_ids,
            "labels": labels,
            "images": image_tensor,
            "image_sizes": image.size,
            "safety_labels": torch.tensor(sample["safety_label"], dtype=torch.long),
        }

    def _mask_non_assistant_tokens(self, labels, prompt):
        marker = "ASSISTANT:"
        pos = prompt.rfind(marker)
        if pos == -1:
            return labels
        prefix = prompt[: pos + len(marker)]
        prefix_ids = self._tokenizer_image_token(
            prefix,
            self.tokenizer,
            self._image_token_index,
            return_tensors="pt",
        ).squeeze(0)
        labels[: prefix_ids.shape[0]] = self.ignore_index
        return labels
