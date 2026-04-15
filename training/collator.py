"""
Data collator for variable-length video sequences in SafeGem-12B training.
"""

import torch
from dataclasses import dataclass
from typing import Optional


@dataclass
class VideoSafetyCollator:
    """Collates variable-length samples into batches.

    Handles:
    - Right-padding input_ids, attention_mask, labels to max batch length
    - Concatenating pixel_values across samples
    - Stacking safety_labels
    - Filtering out None samples (failed video loads)
    """

    pad_token_id: int = 0
    ignore_index: int = -100
    max_length: Optional[int] = None

    def __call__(self, batch: list[dict]) -> dict:
        # Filter out None samples
        batch = [b for b in batch if b is not None]
        if len(batch) == 0:
            return None

        # Determine max sequence length in this batch
        max_len = max(b["input_ids"].shape[0] for b in batch)
        if self.max_length is not None:
            max_len = min(max_len, self.max_length)

        batch_size = len(batch)

        # Pad input_ids, attention_mask, labels
        input_ids = torch.full((batch_size, max_len), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
        labels = torch.full((batch_size, max_len), self.ignore_index, dtype=torch.long)

        for i, b in enumerate(batch):
            seq_len = min(b["input_ids"].shape[0], max_len)
            input_ids[i, :seq_len] = b["input_ids"][:seq_len]
            attention_mask[i, :seq_len] = b["attention_mask"][:seq_len]
            labels[i, :seq_len] = b["labels"][:seq_len]

        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

        # Concatenate pixel_values across the batch
        pv_list = [b["pixel_values"] for b in batch if "pixel_values" in b]
        if pv_list:
            result["pixel_values"] = torch.cat(pv_list, dim=0)

        # Stack safety labels
        safety_labels = torch.stack([b["safety_labels"] for b in batch])
        result["safety_labels"] = safety_labels

        return result
