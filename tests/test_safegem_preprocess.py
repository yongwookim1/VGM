import unittest

import torch

from src.models.safegem.preprocess import prepare_safegem_inputs, resolve_safegem_max_length


class _FakeTokenizer:
    model_max_length = 4096


class _FakeProcessor:
    def __init__(self):
        self.tokenizer = _FakeTokenizer()

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        image_count = sum(1 for item in messages[0]["content"] if item["type"] == "image")
        return f"images={image_count};prompt"

    def __call__(self, text, images, padding=False, return_tensors="pt"):
        seq_len = len(images) * 512 + 100
        return {
            "input_ids": torch.ones((1, seq_len), dtype=torch.long),
            "attention_mask": torch.ones((1, seq_len), dtype=torch.long),
            "pixel_values": torch.ones((1, len(images), 3, 4, 4), dtype=torch.float32),
        }


class SafeGemPreprocessTest(unittest.TestCase):
    def test_resolve_safegem_max_length_uses_tokenizer_limit(self):
        processor = _FakeProcessor()
        self.assertEqual(resolve_safegem_max_length(processor, 8192), 4096)

    def test_prepare_safegem_inputs_reduces_frame_count_to_fit_context(self):
        processor = _FakeProcessor()
        sample = {"question": "q", "answer": "a"}
        frames = [object() for _ in range(8)]

        reduced_frames, inputs, _, max_length = prepare_safegem_inputs(
            processor,
            sample,
            frames,
            max_length=8192,
            include_answer=True,
        )

        self.assertEqual(max_length, 4096)
        self.assertEqual(len(reduced_frames), 7)
        self.assertEqual(inputs["input_ids"].shape[1], 7 * 512 + 100)


if __name__ == "__main__":
    unittest.main()
