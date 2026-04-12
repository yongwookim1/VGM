"""
Data preparation script: merge VideoChatGPT and Video-SafetyBench into a unified
training JSON for SafeQwen2.5-VL video fine-tuning.

Usage:
    python training/prepare_data.py \
        --videochatgpt_dir /home/kyw1654/vsbench/data/videochatgpt \
        --safetybench_dir /home/kyw1654/vsbench/data/video_safetybench \
        --activitynet_video_dir /home/kyw1654/vsbench/videos/activitynet \
        --output_path training/data/train_data.json \
        --languages en ko
"""

import argparse
import json
import os
import random
from pathlib import Path

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

BENIGN_TEMPLATES = [
    "Based on the video, I can provide a general description of what is shown. "
    "The content appears to depict a scene that can be discussed in a safe and "
    "informative manner. Could you clarify what specific aspect you'd like to know about?",
    "Looking at the video content, I can offer an objective and informative response. "
    "The video shows a scene that I can describe factually. What would you like to "
    "know more about?",
    "I can see the video content and am happy to provide a helpful, factual response. "
    "Let me share what I observe and address your question constructively.",
]


def load_videochatgpt(data_dir: str, video_dir: str, languages: list[str]):
    """Load all VideoChatGPT splits and produce unified samples."""
    splits = ["generic_data", "consistency_data", "temporal_data"]
    samples = []

    for split in splits:
        for lang in languages:
            suffix = f"_{lang}" if lang != "en" else ""
            filename = f"{split}{suffix}.json" if lang != "en" else f"{split}.json"
            filepath = os.path.join(data_dir, filename)
            if not os.path.exists(filepath):
                print(f"  [SKIP] {filepath} not found")
                continue

            with open(filepath) as f:
                data = json.load(f)

            for item in data:
                video_name = item["video_name"]
                video_path = os.path.join(video_dir, f"{video_name}.mp4")
                question = item["question"]
                answer = item["answer"]

                samples.append({
                    "dataset": "videochatgpt",
                    "split": split,
                    "question_id": item["question_id"],
                    "video_path": video_path,
                    "question": question,
                    "answer": answer,
                    "safety_label": 0,  # all safe
                    "language": lang,
                })

                # consistency_data may have question_2
                if split == "consistency_data":
                    q2 = item.get("question_2", "None")
                    if q2 and q2 != "None":
                        samples.append({
                            "dataset": "videochatgpt",
                            "split": split,
                            "question_id": item["question_id"] + "_q2",
                            "video_path": video_path,
                            "question": q2,
                            "answer": answer,
                            "safety_label": 0,
                            "language": lang,
                        })

    return samples


def load_safetybench(data_dir: str, languages: list[str]):
    """Load Video-SafetyBench harmful + benign splits."""
    samples = []

    for split_type in ["harmful", "benign"]:
        for lang in languages:
            suffix = f"_{lang}" if lang != "en" else ""
            filename = (
                f"{split_type}_data{suffix}.json"
                if lang != "en"
                else f"{split_type}_data.json"
            )
            filepath = os.path.join(data_dir, filename)
            if not os.path.exists(filepath):
                print(f"  [SKIP] {filepath} not found")
                continue

            with open(filepath) as f:
                data = json.load(f)

            for item in data:
                video_rel_path = item["video_path"]
                video_path = os.path.join(data_dir, video_rel_path)
                subcategory = item.get("subcategory", "")
                category = item.get("category", "")

                if split_type == "harmful":
                    safety_label = get_safety_label(subcategory, category)
                    template = random.choice(REFUSAL_TEMPLATES)
                    sub_display = subcategory.replace("_", " ").lower()
                    answer = template.format(subcategory=sub_display)
                else:
                    safety_label = 0  # benign
                    answer = random.choice(BENIGN_TEMPLATES)

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
                    "language": lang,
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
        default="/home/kyw1654/vsbench/data/videochatgpt",
    )
    parser.add_argument(
        "--safetybench_dir",
        default="/home/kyw1654/vsbench/data/video_safetybench",
    )
    parser.add_argument(
        "--activitynet_video_dir",
        default="/home/kyw1654/vsbench/videos/activitynet",
    )
    parser.add_argument(
        "--output_path",
        default="training/data/train_data.json",
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        default=["en"],
        help="Languages to include: en, ko, or both",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    print("Loading VideoChatGPT data...")
    vcgpt_samples = load_videochatgpt(
        args.videochatgpt_dir, args.activitynet_video_dir, args.languages
    )
    print(f"  Loaded {len(vcgpt_samples)} VideoChatGPT samples")

    print("Loading Video-SafetyBench data...")
    sb_samples = load_safetybench(args.safetybench_dir, args.languages)
    print(f"  Loaded {len(sb_samples)} Video-SafetyBench samples")

    all_samples = vcgpt_samples + sb_samples
    random.shuffle(all_samples)

    print(f"\nTotal samples: {len(all_samples)}")
    print(f"  VideoChatGPT: {len(vcgpt_samples)}")
    print(f"  SafetyBench: {len(sb_samples)}")

    # Safety label distribution
    from collections import Counter
    label_counts = Counter(s["safety_label"] for s in all_samples)
    print(f"\nSafety label distribution:")
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count}")

    # Validate video availability
    print("\nValidating video files...")
    stats = validate_videos(all_samples)
    print(f"  Videos found: {stats['found']}/{stats['total']}")
    if stats["missing"] > 0:
        print(f"  Videos missing: {stats['missing']}")
        print(f"  Sample missing paths:")
        for p in stats["missing_paths_sample"]:
            print(f"    {p}")

    # Save
    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(all_samples, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(all_samples)} samples to {args.output_path}")


if __name__ == "__main__":
    main()
