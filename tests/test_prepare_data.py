import json
import tempfile
import unittest
from pathlib import Path

from src.data.prepare_data import load_safewatch


class PrepareDataTest(unittest.TestCase):
    def test_load_safewatch_manifest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            video = root / "clip.mp4"
            video.write_bytes(b"")
            manifest = root / "train.json"
            manifest.write_text(
                json.dumps(
                    [
                        {
                            "id": "sample-1",
                            "video_path": "clip.mp4",
                            "prompt": "Is this safe?",
                            "label": "safe",
                        }
                    ]
                )
            )

            samples = load_safewatch(str(root), str(manifest))
            self.assertEqual(len(samples), 1)
            self.assertEqual(samples[0]["safety_label"], 0)
            self.assertEqual(samples[0]["question"], "Is this safe?")


if __name__ == "__main__":
    unittest.main()
