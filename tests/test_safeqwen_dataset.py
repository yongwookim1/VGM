import unittest
from unittest.mock import patch

from src.models.safeqwen.dataset import VideoSafetyDataset


class _FakeTokenizer:
    def convert_tokens_to_ids(self, token):
        mapping = {"<|im_start|>": 1, "<|video_pad|>": 2}
        return mapping[token]


class _FakeProcessor:
    tokenizer = _FakeTokenizer()


class SafeQwenDatasetConfigTest(unittest.TestCase):
    def test_rejects_unknown_video_backend(self):
        with self.assertRaises(ValueError):
            VideoSafetyDataset(
                data_path="/tmp/does-not-matter.json",
                processor=_FakeProcessor(),
                video_backend="unknown",
            )

    @patch("src.models.safeqwen.dataset.os.path.isfile", return_value=True)
    @patch("src.models.safeqwen.dataset.json.load", return_value=[{"video_path": "video.mp4"}])
    @patch("builtins.open")
    def test_accepts_standard_video_backend(self, _open_mock, _json_load_mock, _isfile_mock):
        dataset = VideoSafetyDataset(
            data_path="/tmp/train.json",
            processor=_FakeProcessor(),
            video_backend="standard",
        )

        self.assertEqual(dataset.video_backend, "standard")


if __name__ == "__main__":
    unittest.main()
