"""Shared collators for SafeGem and SafeLLaVA training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class SafeGemVideoCollator:
    """Collate SafeGem video/text samples into padded batches."""

    pad_token_id: int = 0
    ignore_index: int = -100
    max_length: Optional[int] = None

    def __call__(self, batch: list[dict]) -> dict | None:
        batch = [b for b in batch if b is not None]
        if not batch:
            return None

        max_len = max(b["input_ids"].shape[0] for b in batch)
        if self.max_length is not None:
            max_len = min(max_len, self.max_length)

        batch_size = len(batch)
        input_ids = torch.full((batch_size, max_len), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
        labels = torch.full((batch_size, max_len), self.ignore_index, dtype=torch.long)

        for i, sample in enumerate(batch):
            seq_len = min(sample["input_ids"].shape[0], max_len)
            input_ids[i, :seq_len] = sample["input_ids"][:seq_len]
            attention_mask[i, :seq_len] = sample["attention_mask"][:seq_len]
            labels[i, :seq_len] = sample["labels"][:seq_len]

        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "safety_labels": torch.stack([sample["safety_labels"] for sample in batch]),
            "num_frames_per_sample": torch.stack([sample["num_frames"] for sample in batch]),
        }
        pixel_values = [sample["pixel_values"] for sample in batch if "pixel_values" in sample]
        if pixel_values:
            first = pixel_values[0]
            if first.dim() == 4:
                result["pixel_values"] = torch.cat(pixel_values, dim=0)
            elif first.dim() == 5:
                result["pixel_values"] = torch.cat(pixel_values, dim=0)
            else:
                raise ValueError(
                    f"Unexpected SafeGem pixel_values rank: {first.dim()}. "
                    "Expected 4D [frames, C, H, W] or 5D [batch, frames, C, H, W]."
                )
        return result


@dataclass
class SafeLLaVAVideoCollator:
    """Collate SafeLLaVA single-frame samples into padded batches."""

    pad_token_id: int = 0
    ignore_index: int = -100
    max_length: Optional[int] = None

    def __call__(self, batch: list[dict]) -> dict | None:
        batch = [b for b in batch if b is not None]
        if not batch:
            return None

        max_len = max(sample["input_ids"].shape[0] for sample in batch)
        if self.max_length is not None:
            max_len = min(max_len, self.max_length)

        batch_size = len(batch)
        input_ids = torch.full((batch_size, max_len), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
        labels = torch.full((batch_size, max_len), self.ignore_index, dtype=torch.long)

        for i, sample in enumerate(batch):
            seq_len = min(sample["input_ids"].shape[0], max_len)
            input_ids[i, :seq_len] = sample["input_ids"][:seq_len]
            attention_mask[i, :seq_len] = 1
            labels[i, :seq_len] = sample["labels"][:seq_len]

        images = torch.stack([sample["images"] for sample in batch])
        image_sizes = [sample["image_sizes"] for sample in batch]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "images": images,
            "image_sizes": image_sizes,
            "safety_labels": torch.stack([sample["safety_labels"] for sample in batch]),
        }
