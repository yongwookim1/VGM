"""
Data collator for variable-length video sequences in SafeQwen2.5-VL training.
"""

import torch
from dataclasses import dataclass
from typing import Optional


@dataclass
class VideoSafetyCollator:
    """Collates variable-length video samples into batches.

    Handles:
    - Right-padding input_ids, attention_mask, labels to max batch length
    - Concatenating pixel_values_videos and video_grid_thw across samples
    - Stacking safety_labels
    - Filtering out None samples (failed video loads)
    """

    pad_token_id: int = 151643  # <|endoftext|>
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

        # Concatenate pixel_values_videos across the batch
        # Each sample's pixel_values_videos: [num_patches, patch_dim]
        # video_grid_thw: [num_videos, 3] (usually [1, 3] per sample)
        pvv_list = [b["pixel_values_videos"] for b in batch if "pixel_values_videos" in b]
        vgt_list = [b["video_grid_thw"] for b in batch if "video_grid_thw" in b]

        if pvv_list:
            result["pixel_values_videos"] = torch.cat(pvv_list, dim=0)
        if vgt_list:
            result["video_grid_thw"] = torch.cat(vgt_list, dim=0)

        # Stack safety labels
        safety_labels = torch.stack([b["safety_labels"] for b in batch])
        result["safety_labels"] = safety_labels

        return result
