import json
import tempfile
import unittest
from pathlib import Path

from src.eval.run_mmlu import extract_prediction_letter, load_mmlu_records, normalize_answer, normalize_record


class MMLUTest(unittest.TestCase):
    def test_normalize_answer(self):
        choices = ["one", "two", "three", "four"]
        self.assertEqual(normalize_answer(2, choices), "C")
        self.assertEqual(normalize_answer("1", choices), "B")
        self.assertEqual(normalize_answer("D", choices), "D")
        self.assertEqual(normalize_answer("two", choices), "B")

    def test_extract_prediction_letter(self):
        self.assertEqual(extract_prediction_letter("Answer: C", 4), "C")
        self.assertEqual(extract_prediction_letter("b", 4), "B")
        self.assertEqual(extract_prediction_letter("The answer is D.", 4), "D")

    def test_load_and_normalize_json_records(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            payload = [
                {
                    "id": "q1",
                    "question": "What is 2 + 2?",
                    "choices": ["1", "2", "3", "4"],
                    "answer": 3,
                    "subject": "math",
                }
            ]
            (root / "mmlu_test.json").write_text(json.dumps(payload))

            records = load_mmlu_records(str(root), "test")
            self.assertEqual(len(records), 1)

            sample = normalize_record(records[0], 0)
            self.assertEqual(sample["question_id"], "q1")
            self.assertEqual(sample["answer"], "D")
            self.assertEqual(sample["choices"][3], "4")


if __name__ == "__main__":
    unittest.main()
