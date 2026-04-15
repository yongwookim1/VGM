"""
Data collator for SafeLLaVA-7B video safety training.
"""

import torch
from dataclasses import dataclass
from typing import Optional

from safellava.constants import IGNORE_INDEX


@dataclass
class VideoSafetyCollator:
    """Collates variable-length samples into batches for SafeLLaVA.

    Handles:
    - Right-padding input_ids and labels to max batch length
    - Stacking image tensors
    - Collecting image_sizes as a list
    - Stacking safety_labels
    - Filtering out None samples (failed video loads)
    """

    pad_token_id: int = 0
    ignore_index: int = IGNORE_INDEX
    max_length: Optional[int] = None

    def __call__(self, batch: list[dict]) -> dict:
        batch = [b for b in batch if b is not None]
        if len(batch) == 0:
            return None

        max_len = max(b["input_ids"].shape[0] for b in batch)
        if self.max_length is not None:
            max_len = min(max_len, self.max_length)

        batch_size = len(batch)

        input_ids = torch.full((batch_size, max_len), self.pad_token_id, dtype=torch.long)
        labels = torch.full((batch_size, max_len), self.ignore_index, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)

        for i, b in enumerate(batch):
            seq_len = min(b["input_ids"].shape[0], max_len)
            input_ids[i, :seq_len] = b["input_ids"][:seq_len]
            labels[i, :seq_len] = b["labels"][:seq_len]
            attention_mask[i, :seq_len] = 1

        # Stack images: each sample has [C, H, W] or [num_patches, C, H, W]
        images = torch.stack([b["image"] for b in batch], dim=0)

        # Collect image sizes as list of [width, height]
        image_sizes = [b["image_size"] for b in batch]

        # Stack safety labels
        safety_labels = torch.stack([b["safety_labels"] for b in batch])

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "images": images,
            "image_sizes": image_sizes,
            "safety_labels": safety_labels,
        }
