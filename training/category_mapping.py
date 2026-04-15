"""Compatibility exports for binary label helpers."""

from src.data.labels import SAFE_LABEL, UNSAFE_LABEL, infer_binary_label, normalize_binary_label


def get_safety_label(subcategory: str, category: str = None) -> int:
    return infer_binary_label(category=category, subcategory=subcategory, default=UNSAFE_LABEL)


__all__ = [
    "SAFE_LABEL",
    "UNSAFE_LABEL",
    "get_safety_label",
    "infer_binary_label",
    "normalize_binary_label",
]
