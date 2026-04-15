import unittest

from src.eval.metrics import compute_binary_metrics


class MetricsTest(unittest.TestCase):
    def test_compute_binary_metrics(self):
        metrics = compute_binary_metrics([0, 1, 1, 0], [0, 1, 0, 0])
        self.assertAlmostEqual(metrics["accuracy"], 0.75)
        self.assertAlmostEqual(metrics["precision"], 1.0)
        self.assertAlmostEqual(metrics["recall"], 0.5)


if __name__ == "__main__":
    unittest.main()
