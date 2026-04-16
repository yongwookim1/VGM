import unittest

import torch

from src.common.collator import SafeGemVideoCollator


class SafeGemCollatorTest(unittest.TestCase):
    def test_flattens_multiframe_pixel_values_across_batch(self):
        collator = SafeGemVideoCollator(pad_token_id=0, ignore_index=-100, max_length=16)
        batch = [
            {
                "input_ids": torch.tensor([1, 2, 3]),
                "attention_mask": torch.tensor([1, 1, 1]),
                "labels": torch.tensor([-100, 2, 3]),
                "safety_labels": torch.tensor(1),
                "num_frames": torch.tensor(4),
                "pixel_values": torch.zeros(4, 3, 8, 8),
            },
            {
                "input_ids": torch.tensor([4, 5]),
                "attention_mask": torch.tensor([1, 1]),
                "labels": torch.tensor([-100, 5]),
                "safety_labels": torch.tensor(0),
                "num_frames": torch.tensor(4),
                "pixel_values": torch.ones(4, 3, 8, 8),
            },
        ]

        output = collator(batch)

        self.assertEqual(output["pixel_values"].shape, (8, 3, 8, 8))
        self.assertEqual(output["num_frames_per_sample"].tolist(), [4, 4])


if __name__ == "__main__":
    unittest.main()
