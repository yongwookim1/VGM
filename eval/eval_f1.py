"""
F1 evaluation using ground-truth safety labels vs. safety head predictions.

Usage:
    python eval/eval_f1.py eval/results/predictions.json
"""

import sys
import json
import argparse
from collections import defaultdict


def compute_metrics(y_true, y_pred):
    tp = sum(p == 1 and t == 1 for t, p in zip(y_true, y_pred))
    fp = sum(p == 1 and t == 0 for t, p in zip(y_true, y_pred))
    tn = sum(p == 0 and t == 0 for t, p in zip(y_true, y_pred))
    fn = sum(p == 0 and t == 1 for t, p in zip(y_true, y_pred))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy  = (tp + tn) / len(y_true) if y_true else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file")
    args = parser.parse_args()

    with open(args.input_file) as f:
        data = json.load(f)

    # Drop samples where inference failed, safety_pred is missing, or label is -100
    # (safety_label=-100 means the subcategory has no HoliSafe mapping — excluded from loss)
    valid = [
        d for d in data
        if d.get("safety_pred") is not None
        and d.get("safety_label", -100) != -100
    ]
    skipped = len(data) - len(valid)
    if skipped:
        print(f"Skipped {skipped} samples (missing pred or safety_label=-100)\n")

    # safety_label is a 20-class index: 0=safe, 1-19=unsafe category.
    # Binarize: 0 -> safe (0), anything else -> unsafe (1).
    y_true = [0 if d["safety_label"] == 0 else 1 for d in valid]
    y_pred = [0 if d["safety_pred"] == 0 else 1 for d in valid]

    # Overall metrics
    m = compute_metrics(y_true, y_pred)
    print(f"{'='*45}")
    print(f"  Overall  ({len(valid)} samples)")
    print(f"{'='*45}")
    print(f"  Accuracy : {m['accuracy']:.4f}")
    print(f"  Precision: {m['precision']:.4f}")
    print(f"  Recall   : {m['recall']:.4f}")
    print(f"  F1       : {m['f1']:.4f}")
    print(f"  TP/FP/TN/FN: {m['tp']}/{m['fp']}/{m['tn']}/{m['fn']}")

    # Per-category breakdown (if subcategory field present)
    by_cat = defaultdict(lambda: {"true": [], "pred": []})
    for d in valid:
        cat = d.get("subcategory") or d.get("category") or "unknown"
        # Binarize here too — same as overall
        by_cat[cat]["true"].append(0 if d["safety_label"] == 0 else 1)
        by_cat[cat]["pred"].append(0 if d["safety_pred"] == 0 else 1)

    if len(by_cat) > 1:
        print(f"\n{'='*45}")
        print(f"  Per-category F1")
        print(f"{'='*45}")
        print(f"  {'Category':<35} {'N':>5}  {'F1':>6}")
        print(f"  {'-'*35}  {'-'*5}  {'-'*6}")
        for cat in sorted(by_cat):
            cm = compute_metrics(by_cat[cat]["true"], by_cat[cat]["pred"])
            n = len(by_cat[cat]["true"])
            print(f"  {cat:<35} {n:>5}  {cm['f1']:>6.4f}")


if __name__ == "__main__":
    main()
