"""Run local MMLU evaluation for text generation models."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.common.io import write_json

try:
    from datasets import load_dataset, load_from_disk
except ImportError:  # pragma: no cover - optional dependency
    load_dataset = None
    load_from_disk = None


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

OPTION_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
ANSWER_RE = re.compile(r"\b([A-Z])\b")


def load_json(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        payload = json.load(f)
    if isinstance(payload, dict):
        for key in ("data", "examples", "rows"):
            if isinstance(payload.get(key), list):
                return payload[key]
        return [payload]
    return payload


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def load_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def load_parquet(path: Path) -> list[dict[str, Any]]:
    if load_dataset is None:
        raise RuntimeError("datasets is required to read parquet MMLU files")
    dataset = load_dataset("parquet", data_files=str(path), split="train")
    return list(dataset)


def find_candidate_files(path: Path, split: str) -> list[Path]:
    if path.is_file():
        return [path]

    patterns = [
        f"**/*{split}*.json",
        f"**/*{split}*.jsonl",
        f"**/*{split}*.csv",
        f"**/*{split}*.parquet",
    ]
    files: list[Path] = []
    for pattern in patterns:
        files.extend(sorted(path.glob(pattern)))
    if files:
        return files

    fallback_patterns = ["**/*.json", "**/*.jsonl", "**/*.csv", "**/*.parquet"]
    for pattern in fallback_patterns:
        files.extend(sorted(path.glob(pattern)))
    return files


def load_mmlu_records(path_str: str, split: str) -> list[dict[str, Any]]:
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"MMLU path does not exist: {path}")

    if path.is_dir() and load_from_disk is not None:
        try:
            dataset_obj = load_from_disk(str(path))
            if split in dataset_obj:
                return list(dataset_obj[split])
        except Exception:
            pass

    records: list[dict[str, Any]] = []
    for candidate in find_candidate_files(path, split):
        suffix = candidate.suffix.lower()
        if suffix == ".json":
            records.extend(load_json(candidate))
        elif suffix == ".jsonl":
            records.extend(load_jsonl(candidate))
        elif suffix == ".csv":
            records.extend(load_csv(candidate))
        elif suffix == ".parquet":
            records.extend(load_parquet(candidate))

    if not records:
        raise FileNotFoundError(f"No MMLU records found under {path}")
    return records


def extract_choices(record: dict[str, Any]) -> list[str]:
    raw_choices = None
    for key in ("choices", "options", "answer_choices"):
        if key in record:
            raw_choices = record[key]
            break

    if raw_choices is None:
        option_keys = [key for key in ("A", "B", "C", "D", "E") if key in record]
        if option_keys:
            return [str(record[key]).strip() for key in option_keys]
        raise KeyError("No answer choices found in MMLU record")

    if isinstance(raw_choices, dict):
        letter_keys = [key for key in OPTION_LETTERS if key in raw_choices]
        if letter_keys:
            return [str(raw_choices[key]).strip() for key in letter_keys]
        return [str(value).strip() for _, value in sorted(raw_choices.items())]

    choices: list[str] = []
    for item in raw_choices:
        if isinstance(item, dict):
            text = item.get("text", item.get("content", item.get("label", "")))
            choices.append(str(text).strip())
        else:
            choices.append(str(item).strip())
    return choices


def normalize_answer(answer: Any, choices: list[str]) -> str | None:
    if answer is None:
        return None

    if isinstance(answer, int):
        return OPTION_LETTERS[answer] if 0 <= answer < len(choices) else None

    answer_str = str(answer).strip()
    if not answer_str:
        return None

    if answer_str.isdigit():
        index = int(answer_str)
        return OPTION_LETTERS[index] if 0 <= index < len(choices) else None

    upper = answer_str.upper()
    if upper in OPTION_LETTERS[: len(choices)]:
        return upper

    match = ANSWER_RE.search(upper)
    if match and match.group(1) in OPTION_LETTERS[: len(choices)]:
        return match.group(1)

    for idx, choice in enumerate(choices):
        if answer_str == choice or answer_str.lower() == choice.lower():
            return OPTION_LETTERS[idx]
    return None


def normalize_record(record: dict[str, Any], index: int) -> dict[str, Any]:
    question = None
    for key in ("question", "input", "problem", "prompt", "query"):
        if key in record and record[key]:
            question = str(record[key]).strip()
            break
    if not question:
        raise KeyError("No question field found in MMLU record")

    choices = extract_choices(record)
    answer = None
    for key in (
        "answer",
        "target",
        "label",
        "gold",
        "correct",
        "answer_idx",
        "answer_index",
        "target_idx",
    ):
        if key in record:
            answer = normalize_answer(record[key], choices)
            if answer is not None:
                break

    if answer is None:
        raise KeyError("No valid answer field found in MMLU record")

    subject = ""
    for key in ("subject", "category", "domain"):
        if key in record and record[key]:
            subject = str(record[key]).strip()
            break

    return {
        "question_id": str(record.get("id", record.get("question_id", index))),
        "question": question,
        "choices": choices,
        "answer": answer,
        "subject": subject,
    }


def build_prompt(sample: dict[str, Any]) -> str:
    lines = []
    if sample["subject"]:
        lines.append(f"The following is a multiple choice question about {sample['subject']}.")
    else:
        lines.append("The following is a multiple choice question.")
    lines.append("")
    lines.append(sample["question"])
    lines.append("")
    for idx, choice in enumerate(sample["choices"]):
        lines.append(f"{OPTION_LETTERS[idx]}. {choice}")
    lines.append("")
    lines.append("Answer with only the option letter.")
    lines.append("Answer:")
    return "\n".join(lines)


def extract_prediction_letter(text: str, num_choices: int) -> str | None:
    valid = set(OPTION_LETTERS[:num_choices])
    stripped = text.strip().upper()
    if stripped in valid:
        return stripped
    for match in ANSWER_RE.findall(stripped):
        if match in valid:
            return match
    return None


def load_model_and_tokenizer(args):
    from peft import PeftModel

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        ignore_mismatched_sizes=True,
    )

    if args.model_path and not args.no_lora:
        logger.info("Loading LoRA adapter from %s", args.model_path)
        model = PeftModel.from_pretrained(model, args.model_path)
        model = model.merge_and_unload()

    model.eval()
    model.to(args.device)
    return model, tokenizer


@torch.inference_mode()
def generate_answer_letter(model, tokenizer, sample: dict[str, Any], args) -> tuple[str, str]:
    prompt = build_prompt(sample)
    inputs = tokenizer(prompt, return_tensors="pt").to(args.device)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    prompt_len = inputs["input_ids"].shape[1]
    generated_ids = output_ids[0][prompt_len:]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return text, extract_prediction_letter(text, len(sample["choices"])) or ""


def compute_accuracy(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    correct = sum(1 for row in rows if row.get("predicted_answer") == row.get("answer"))
    accuracy = correct / total if total else 0.0
    by_subject: dict[str, dict[str, Any]] = {}
    for row in rows:
        subject = row.get("subject") or "unknown"
        stats = by_subject.setdefault(subject, {"total": 0, "correct": 0})
        stats["total"] += 1
        stats["correct"] += int(row.get("predicted_answer") == row.get("answer"))
    for stats in by_subject.values():
        stats["accuracy"] = stats["correct"] / stats["total"] if stats["total"] else 0.0
    return {
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "by_subject": by_subject,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MMLU evaluation on a local dataset copy")
    parser.add_argument("--model_type", required=True, choices=["safegem", "safellava"])
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--mmlu_path", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--metrics_file", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_new_tokens", type=int, default=8)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--model_path", default="")
    parser.add_argument("--no_lora", action="store_true")
    args = parser.parse_args()

    raw_records = load_mmlu_records(args.mmlu_path, args.split)
    samples = [normalize_record(record, index) for index, record in enumerate(raw_records)]
    if args.max_samples > 0:
        samples = samples[: args.max_samples]

    logger.info("Loaded %s MMLU samples from %s", len(samples), args.mmlu_path)
    model, tokenizer = load_model_and_tokenizer(args)

    rows: list[dict[str, Any]] = []
    for sample in tqdm(samples, desc=f"{args.model_type} MMLU"):
        generated_text, predicted_answer = generate_answer_letter(model, tokenizer, sample, args)
        row = dict(sample)
        row["generated_text"] = generated_text
        row["predicted_answer"] = predicted_answer
        row["is_correct"] = predicted_answer == sample["answer"]
        rows.append(row)

        if len(rows) % 100 == 0:
            write_json(args.output_file, rows)

    metrics = compute_accuracy(rows)
    write_json(args.output_file, rows)
    write_json(args.metrics_file, metrics)
    logger.info("MMLU accuracy: %.4f (%s/%s)", metrics["accuracy"], metrics["correct"], metrics["total"])


if __name__ == "__main__":
    main()
