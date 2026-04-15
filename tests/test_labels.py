import unittest

from src.data.labels import SAFE_LABEL, UNSAFE_LABEL, infer_binary_label, normalize_binary_label


class LabelsTest(unittest.TestCase):
    def test_normalize_binary_label(self):
        self.assertEqual(normalize_binary_label("safe"), SAFE_LABEL)
        self.assertEqual(normalize_binary_label("harmful"), UNSAFE_LABEL)
        self.assertEqual(normalize_binary_label(True), 1)

    def test_infer_binary_label_from_split(self):
        self.assertEqual(infer_binary_label(split="benign"), SAFE_LABEL)
        self.assertEqual(infer_binary_label(split="harmful"), UNSAFE_LABEL)


if __name__ == "__main__":
    unittest.main()
