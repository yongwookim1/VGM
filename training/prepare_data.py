"""
Data preparation script: merge VideoChatGPT and Video-SafetyBench into a unified
training JSON for SafeQwen2.5-VL video fine-tuning.

Usage:
    python training/prepare_data.py \
        --videochatgpt_dir data/videochatgpt \
        --safetybench_dir data/video_safetybench \
        --output_path training/data/train_data.json
"""

import argparse
import json
import os
import random
from collections import Counter

from category_mapping import get_safety_label

REFUSAL_TEMPLATES = [
    "I'm sorry, but I cannot assist with that request. The video contains content "
    "related to {subcategory}, and providing guidance on harmful activities goes "
    "against my safety guidelines.",
    "I cannot help with this request as it involves potentially harmful or unsafe "
    "content. If you have a safe and constructive question, I'm happy to assist.",
    "This request asks me to engage with harmful content. I must decline. Please "
    "let me know if there's something safe and constructive I can help you with.",
    "I'm not able to provide assistance with this type of request. The content "
    "appears to involve {subcategory}, which raises safety concerns. I'd be glad "
    "to help with other questions instead.",
    "I appreciate your question, but I'm unable to assist with requests involving "
    "potentially harmful content. Please feel free to ask me about something else.",
]


def load_videochatgpt(data_dir: str):
    """Load all VideoChatGPT splits. Videos are expected inside data_dir."""
    splits = ["generic_data", "consistency_data", "temporal_data"]
    samples = []

    for split in splits:
        filepath = os.path.join(data_dir, f"{split}.json")
        if not os.path.exists(filepath):
            print(f"  [SKIP] {filepath} not found")
            continue

        with open(filepath) as f:
            data = json.load(f)

        for item in data:
            video_name = item["video_name"]
            video_path = os.path.join(data_dir, f"{video_name}.mp4")

            samples.append({
                "dataset": "videochatgpt",
                "split": split,
                "question_id": item["question_id"],
                "video_path": video_path,
                "question": item["question"],
                "answer": item["answer"],
                "safety_label": 0,  # all safe
            })

            if split == "consistency_data":
                q2 = item.get("question_2", "None")
                if q2 and q2 != "None":
                    samples.append({
                        "dataset": "videochatgpt",
                        "split": split,
                        "question_id": item["question_id"] + "_q2",
                        "video_path": video_path,
                        "question": q2,
                        "answer": item["answer"],
                        "safety_label": 0,
                    })

    return samples


def load_safetybench(data_dir: str):
    """Load Video-SafetyBench harmful + benign splits."""
    samples = []

    for split_type in ["harmful", "benign"]:
        filepath = os.path.join(data_dir, f"{split_type}_data.json")
        if not os.path.exists(filepath):
            print(f"  [SKIP] {filepath} not found")
            continue

        with open(filepath) as f:
            data = json.load(f)

        for item in data:
            video_path = os.path.join(data_dir, item["video_path"])
            subcategory = item.get("subcategory", "")
            category = item.get("category", "")

            sub_display = subcategory.replace("_", " ").lower()
            answer = random.choice(REFUSAL_TEMPLATES).format(subcategory=sub_display)
            safety_label = 1  # both splits contain unsafe videos

            samples.append({
                "dataset": f"safetybench_{split_type}",
                "split": split_type,
                "question_id": item["question_id"],
                "video_path": video_path,
                "question": item["question"],
                "answer": answer,
                "safety_label": safety_label,
                "category": category,
                "subcategory": subcategory,
            })

    return samples


def validate_videos(samples: list[dict]) -> dict:
    """Check which video files actually exist."""
    total = len(samples)
    found = sum(1 for s in samples if os.path.isfile(s["video_path"]))
    missing_paths = set(
        s["video_path"] for s in samples if not os.path.isfile(s["video_path"])
    )
    return {
        "total": total,
        "found": found,
        "missing": total - found,
        "missing_paths_sample": list(missing_paths)[:10],
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare unified training data")
    parser.add_argument(
        "--videochatgpt_dir",
        default="data/videochatgpt",
    )
    parser.add_argument(
        "--safetybench_dir",
        default="data/video_safetybench",
    )
    parser.add_argument(
        "--output_path",
        default="training/data/train_data.json",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.2,
        help="Fraction of data to hold out for testing (0 to skip split)",
    )
    args = parser.parse_args()

    random.seed(args.seed)

    print("Loading VideoChatGPT data...")
    vcgpt_samples = load_videochatgpt(args.videochatgpt_dir)
    print(f"  Loaded {len(vcgpt_samples)} samples")

    print("Loading Video-SafetyBench data...")
    sb_samples = load_safetybench(args.safetybench_dir)
    print(f"  Loaded {len(sb_samples)} samples")

    all_samples = vcgpt_samples + sb_samples
    random.shuffle(all_samples)

    print(f"\nTotal: {len(all_samples)} (VideoChatGPT: {len(vcgpt_samples)}, SafetyBench: {len(sb_samples)})")

    label_counts = Counter(s["safety_label"] for s in all_samples)
    print(f"\nSafety label distribution:")
    for label, count in sorted(label_counts.items()):
        print(f"  {'safe' if label == 0 else 'unsafe'} ({label}): {count}")

    print("\nValidating video files...")
    stats = validate_videos(all_samples)
    print(f"  Videos found: {stats['found']}/{stats['total']}")
    if stats["missing"] > 0:
        print(f"  Videos missing: {stats['missing']}")
        for p in stats["missing_paths_sample"]:
            print(f"    {p}")

    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    if args.test_ratio > 0:
        split_idx = int(len(all_samples) * (1 - args.test_ratio))
        train_samples = all_samples[:split_idx]
        test_samples = all_samples[split_idx:]

        test_path = args.output_path.replace("train_data", "test_data")
        with open(args.output_path, "w") as f:
            json.dump(train_samples, f, indent=2, ensure_ascii=False)
        with open(test_path, "w") as f:
            json.dump(test_samples, f, indent=2, ensure_ascii=False)
        print(f"\nSaved {len(train_samples)} train / {len(test_samples)} test samples")
    else:
        with open(args.output_path, "w") as f:
            json.dump(all_samples, f, indent=2, ensure_ascii=False)
        print(f"\nSaved {len(all_samples)} samples to {args.output_path}")


if __name__ == "__main__":
    main()
