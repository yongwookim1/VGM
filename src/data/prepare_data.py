"""Prepare a unified train/test JSON from supported video safety datasets."""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter
from pathlib import Path
from typing import Iterable

from src.common.io import write_json
from src.data.labels import SAFE_LABEL, UNSAFE_LABEL, infer_binary_label

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

SAFE_RESPONSE_TEMPLATES = [
    "The video appears safe to discuss. How can I help you with it?",
    "This looks like benign content. Let me know what you would like to understand about the video.",
]


def find_video(video_dir: str, video_name: str) -> str:
    """Find a video file by name, trying multiple extensions."""
    for ext in (".mkv", ".mp4", ".avi", ".webm", ".mov"):
        path = os.path.join(video_dir, f"{video_name}{ext}")
        if os.path.isfile(path):
            return path
    return os.path.join(video_dir, f"{video_name}.mp4")


def load_json_or_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    if path.suffix == ".jsonl":
        with open(path) as f:
            return [json.loads(line) for line in f if line.strip()]

    with open(path) as f:
        payload = json.load(f)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("data", "samples", "items", "annotations"):
            if isinstance(payload.get(key), list):
                return payload[key]
    raise ValueError(f"Unsupported manifest structure: {path}")


def derive_test_output_path(output_path: str | Path) -> Path:
    output_path = Path(output_path)
    if "train_data" in output_path.name:
        return output_path.with_name(output_path.name.replace("train_data", "test_data"))
    return output_path.with_name(f"{output_path.stem}_test{output_path.suffix}")


def load_videochatgpt(data_dir: str) -> list[dict]:
    """Load VideoChatGPT splits from the standard release layout."""
    video_dir = os.path.join(data_dir, "Test_Videos")
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
            video_path = find_video(video_dir, video_name)
            samples.append(
                {
                    "dataset": "videochatgpt",
                    "split": split,
                    "question_id": item["question_id"],
                    "video_path": video_path,
                    "question": item["question"],
                    "answer": item["answer"],
                    "safety_label": SAFE_LABEL,
                }
            )

            if split == "consistency_data":
                q2 = item.get("question_2", "None")
                if q2 and q2 != "None":
                    samples.append(
                        {
                            "dataset": "videochatgpt",
                            "split": split,
                            "question_id": item["question_id"] + "_q2",
                            "video_path": video_path,
                            "question": q2,
                            "answer": item["answer"],
                            "safety_label": SAFE_LABEL,
                        }
                    )

    return samples


def build_refusal_answer(subcategory: str) -> str:
    text = (subcategory or "harmful content").replace("_", " ").lower()
    return random.choice(REFUSAL_TEMPLATES).format(subcategory=text)


def build_safe_answer() -> str:
    return random.choice(SAFE_RESPONSE_TEMPLATES)


def load_safetybench(data_dir: str) -> list[dict]:
    """Load Video-SafetyBench harmful/benign splits.

    In this training setup, all Video-SafetyBench samples are treated as unsafe.
    The split names are preserved for analysis, but both map to label 1.
    """
    samples = []

    for split_type in ("harmful", "benign"):
        filepath = os.path.join(data_dir, f"{split_type}_data.json")
        if not os.path.exists(filepath):
            print(f"  [SKIP] {filepath} not found")
            continue

        with open(filepath) as f:
            data = json.load(f)

        for item in data:
            video_path = os.path.join(data_dir, item["video_path"])
            category = item.get("category", "")
            subcategory = item.get("subcategory", "")
            safety_label = UNSAFE_LABEL
            answer = build_refusal_answer(subcategory)
            samples.append(
                {
                    "dataset": "videosafetybench",
                    "split": split_type,
                    "question_id": item["question_id"],
                    "video_path": video_path,
                    "question": item["question"],
                    "answer": answer,
                    "safety_label": safety_label,
                    "category": category,
                    "subcategory": subcategory,
                }
            )

    return samples


def discover_safewatch_manifests(safewatch_dir: str | None, manifest_path: str | None) -> list[str]:
    if manifest_path:
        return [manifest_path]
    if not safewatch_dir:
        return []

    root = Path(safewatch_dir)
    candidates = []
    preferred_names = [
        "train.json",
        "train.jsonl",
        "test.json",
        "test.jsonl",
        "val.json",
        "val.jsonl",
        "annotations.json",
        "annotations.jsonl",
        "metadata.json",
    ]
    for name in preferred_names:
        path = root / name
        if path.exists():
            candidates.append(str(path))
    if candidates:
        return candidates

    return [str(path) for path in sorted(root.rglob("*.json"))] + [
        str(path) for path in sorted(root.rglob("*.jsonl"))
    ]


def get_first(record: dict, keys: Iterable[str], default=None):
    for key in keys:
        if key in record and record[key] not in (None, ""):
            return record[key]
    return default


def resolve_video_path(record: dict, root_dir: str | None) -> str:
    raw_path = get_first(record, ("video_path", "video", "path", "file_path", "video_file"))
    if raw_path is None:
        raise KeyError("SafeWatch sample is missing a video path field")

    raw_path = str(raw_path)
    if os.path.isabs(raw_path) or root_dir is None:
        return raw_path
    return os.path.join(root_dir, raw_path)


def load_safewatch(safewatch_dir: str | None = None, manifest_path: str | None = None) -> list[dict]:
    """Load SafeWatch manifests with flexible field mapping.

    This loader accepts either an explicit manifest file or a dataset directory.
    It supports common field aliases so you can point it at a SafeWatch export
    without patching code on the target server.
    """
    manifests = discover_safewatch_manifests(safewatch_dir, manifest_path)
    samples = []
    for manifest in manifests:
        split_hint = Path(manifest).stem.lower()
        for idx, item in enumerate(load_json_or_jsonl(manifest)):
            category = get_first(item, ("category", "policy", "domain"), "")
            subcategory = get_first(item, ("subcategory", "sub_category", "topic"), "")
            safety_label = infer_binary_label(
                explicit_label=get_first(
                    item,
                    ("safety_label", "label", "binary_label", "is_unsafe", "harmfulness"),
                ),
                split=get_first(item, ("split", "subset"), split_hint),
                category=category,
                subcategory=subcategory,
                default=UNSAFE_LABEL,
            )
            question = get_first(item, ("question", "prompt", "query", "instruction"), "")
            if not question:
                raise KeyError(f"SafeWatch sample in {manifest} is missing a question field")
            answer = get_first(
                item,
                ("answer", "response", "assistant_response", "output", "chosen"),
            )
            if not answer:
                answer = (
                    build_refusal_answer(subcategory)
                    if safety_label == UNSAFE_LABEL
                    else build_safe_answer()
                )
            question_id = get_first(item, ("question_id", "id", "sample_id"), f"{Path(manifest).stem}_{idx}")
            samples.append(
                {
                    "dataset": "safewatch",
                    "split": get_first(item, ("split", "subset"), split_hint),
                    "question_id": str(question_id),
                    "video_path": resolve_video_path(item, safewatch_dir),
                    "question": question,
                    "answer": answer,
                    "safety_label": safety_label,
                    "category": category,
                    "subcategory": subcategory,
                }
            )
    return samples


def validate_videos(samples: list[dict]) -> dict:
    total = len(samples)
    found = sum(1 for sample in samples if os.path.isfile(sample["video_path"]))
    missing_paths = sorted(
        {sample["video_path"] for sample in samples if not os.path.isfile(sample["video_path"])}
    )
    return {
        "total": total,
        "found": found,
        "missing": total - found,
        "missing_paths_sample": missing_paths[:10],
    }


def maybe_split(samples: list[dict], test_ratio: float) -> tuple[list[dict], list[dict]]:
    if test_ratio <= 0:
        return samples, []
    split_idx = int(len(samples) * (1 - test_ratio))
    return samples[:split_idx], samples[split_idx:]


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare unified video safety data")
    parser.add_argument("--videochatgpt_dir", default=None)
    parser.add_argument("--safetybench_dir", default=None)
    parser.add_argument("--safewatch_dir", default=None)
    parser.add_argument("--safewatch_manifest", default=None)
    parser.add_argument("--output_path", default="data/processed/train_data.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_ratio", type=float, default=0.2)
    args = parser.parse_args()

    random.seed(args.seed)

    sources = []
    if args.videochatgpt_dir:
        print("Loading VideoChatGPT data...")
        vcgpt_samples = load_videochatgpt(args.videochatgpt_dir)
        print(f"  Loaded {len(vcgpt_samples)} samples")
        sources.extend(vcgpt_samples)

    if args.safetybench_dir:
        print("Loading Video-SafetyBench data...")
        safetybench_samples = load_safetybench(args.safetybench_dir)
        print(f"  Loaded {len(safetybench_samples)} samples")
        sources.extend(safetybench_samples)

    if args.safewatch_dir or args.safewatch_manifest:
        print("Loading SafeWatch data...")
        safewatch_samples = load_safewatch(args.safewatch_dir, args.safewatch_manifest)
        print(f"  Loaded {len(safewatch_samples)} samples")
        sources.extend(safewatch_samples)

    if not sources:
        raise SystemExit(
            "No data source configured. Pass at least one of "
            "--videochatgpt_dir, --safetybench_dir, or --safewatch_dir/--safewatch_manifest."
        )

    random.shuffle(sources)
    label_counts = Counter(sample["safety_label"] for sample in sources)

    print(f"\nTotal samples: {len(sources)}")
    print("Safety label distribution:")
    for label, count in sorted(label_counts.items()):
        label_name = "safe" if label == SAFE_LABEL else "unsafe"
        print(f"  {label_name} ({label}): {count}")

    print("\nValidating video files...")
    stats = validate_videos(sources)
    print(f"  Videos found: {stats['found']}/{stats['total']}")
    if stats["missing"] > 0:
        print(f"  Videos missing: {stats['missing']}")
        for path in stats["missing_paths_sample"]:
            print(f"    {path}")

    train_samples, test_samples = maybe_split(sources, args.test_ratio)
    output_path = Path(args.output_path)
    write_json(output_path, train_samples)
    print(f"\nSaved {len(train_samples)} train samples to {output_path}")

    if test_samples:
        test_path = derive_test_output_path(output_path)
        write_json(test_path, test_samples)
        print(f"Saved {len(test_samples)} test samples to {test_path}")


if __name__ == "__main__":
    main()
