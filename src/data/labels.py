"""Binary label helpers shared across dataset preparation and evaluation."""

from __future__ import annotations

from typing import Any

SAFE_LABEL = 0
UNSAFE_LABEL = 1

SAFE_STRINGS = {
    "0",
    "safe",
    "benign",
    "harmless",
    "non-harmful",
    "non_harmful",
    "unharmful",
}
UNSAFE_STRINGS = {
    "1",
    "unsafe",
    "harmful",
    "toxic",
    "dangerous",
}


def normalize_binary_label(value: Any) -> int:
    """Normalize a loosely formatted binary label to 0/1."""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        if value in (SAFE_LABEL, UNSAFE_LABEL):
            return value
        raise ValueError(f"Unsupported integer label: {value}")

    text = str(value).strip().lower()
    if text in SAFE_STRINGS:
        return SAFE_LABEL
    if text in UNSAFE_STRINGS:
        return UNSAFE_LABEL
    raise ValueError(f"Unsupported binary label value: {value!r}")


def infer_binary_label(
    *,
    explicit_label: Any = None,
    split: str | None = None,
    category: str | None = None,
    subcategory: str | None = None,
    default: int | None = None,
) -> int:
    """Infer a binary label from explicit labels or dataset metadata.

    SafeWatch and Video-SafetyBench commonly provide either an explicit binary
    label or split/category metadata. This helper keeps the repo on one label
    contract: safe=0, unsafe=1.
    """
    if explicit_label is not None and explicit_label != "":
        return normalize_binary_label(explicit_label)

    normalized_split = (split or "").strip().lower()
    if normalized_split in {"safe", "benign", "normal"}:
        return SAFE_LABEL
    if normalized_split in {"unsafe", "harmful"}:
        return UNSAFE_LABEL

    joined = " ".join(
        part.strip().lower()
        for part in (category or "", subcategory or "")
        if part
    )
    if any(token in joined for token in ("safe", "benign", "normal")):
        return SAFE_LABEL
    if joined:
        return UNSAFE_LABEL

    if default is not None:
        return default
    raise ValueError(
        "Could not infer a binary safety label from the provided sample metadata."
    )
