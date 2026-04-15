"""Evaluate prediction files against the shared binary safety label contract."""

from __future__ import annotations

import argparse
from collections import defaultdict

from src.common.io import load_json, write_json
from src.eval.metrics import compute_binary_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file")
    args = parser.parse_args()

    data = load_json(args.input_file)
    valid = [
        row
        for row in data
        if row.get("safety_pred") is not None and row.get("safety_label", -100) != -100
    ]
    skipped = len(data) - len(valid)
    if skipped:
        print(f"Skipped {skipped} samples (missing pred or safety_label=-100)\n")

    y_true = [row["safety_label"] for row in valid]
    y_pred = [0 if row["safety_pred"] == 0 else 1 for row in valid]

    overall = compute_binary_metrics(y_true, y_pred)
    print(f"{'=' * 45}")
    print(f"  Overall ({len(valid)} samples)")
    print(f"{'=' * 45}")
    print(f"  Accuracy : {overall['accuracy']:.4f}")
    print(f"  Precision: {overall['precision']:.4f}")
    print(f"  Recall   : {overall['recall']:.4f}")
    print(f"  F1       : {overall['f1']:.4f}")
    print(
        f"  TP/FP/TN/FN: {overall['tp']}/{overall['fp']}/{overall['tn']}/{overall['fn']}"
    )

    by_cat = defaultdict(lambda: {"true": [], "pred": []})
    for row in valid:
        category = row.get("subcategory") or row.get("category") or "unknown"
        by_cat[category]["true"].append(0 if row["safety_label"] == 0 else 1)
        by_cat[category]["pred"].append(0 if row["safety_pred"] == 0 else 1)

    metrics_payload = {"overall": {**overall, "n": len(valid)}}
    if len(by_cat) > 1:
        print(f"\n{'=' * 45}")
        print("  Per-category F1")
        print(f"{'=' * 45}")
        print(f"  {'Category':<35} {'N':>5}  {'F1':>6}")
        print(f"  {'-' * 35}  {'-' * 5}  {'-' * 6}")

        per_category = {}
        for category in sorted(by_cat):
            category_metrics = compute_binary_metrics(
                by_cat[category]["true"],
                by_cat[category]["pred"],
            )
            n = len(by_cat[category]["true"])
            print(f"  {category:<35} {n:>5}  {category_metrics['f1']:>6.4f}")
            per_category[category] = {**category_metrics, "n": n}

        metrics_payload["per_category"] = per_category

    metrics_path = args.input_file.replace(".json", "_metrics.json")
    write_json(metrics_path, metrics_payload)
    print(f"\nMetrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
