"""Shared SafeGem preprocessing helpers for training and inference."""

from __future__ import annotations

from typing import Any


def resolve_safegem_max_length(processor, requested_max_length: int) -> int:
    tokenizer_max_length = getattr(processor.tokenizer, "model_max_length", None)
    if tokenizer_max_length is None:
        return requested_max_length
    if tokenizer_max_length > 1_000_000:
        return requested_max_length
    return min(requested_max_length, int(tokenizer_max_length))


def build_messages(sample: dict[str, Any], num_images: int, include_answer: bool) -> list[dict[str, Any]]:
    user_content = [{"type": "image"} for _ in range(num_images)]
    user_content.append({"type": "text", "text": sample["question"]})
    messages = [{"role": "user", "content": user_content}]
    if include_answer:
        messages.append(
            {"role": "assistant", "content": [{"type": "text", "text": sample["answer"]}]}
        )
    return messages


def prepare_safegem_inputs(
    processor,
    sample: dict[str, Any],
    frames,
    max_length: int,
    include_answer: bool,
):
    """Tokenize SafeGem inputs while shrinking frame count to fit the context budget."""
    max_length = resolve_safegem_max_length(processor, max_length)
    current_frames = list(frames)
    if not current_frames:
        raise ValueError("SafeGem preprocessing requires at least one frame")

    while current_frames:
        messages = build_messages(sample, len(current_frames), include_answer=include_answer)
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=not include_answer,
        )
        inputs = processor(
            text=[text],
            images=current_frames,
            padding=False,
            return_tensors="pt",
        )
        seq_len = inputs["input_ids"].shape[1]
        if seq_len <= max_length:
            return current_frames, inputs, text, max_length

        if len(current_frames) == 1:
            break
        current_frames = current_frames[:-1]

    raise ValueError(
        f"SafeGem tokenized length exceeds max_length={max_length} even with one frame"
    )
