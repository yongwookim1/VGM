"""
Binary safety label mapping for Video-SafetyBench.

Labels:
  0: safe
  1: unsafe
"""

HOLISAFE_CATEGORIES = ["safe", "unsafe"]


def get_safety_label(subcategory: str, category: str = None) -> int:
    """Get the binary safety label for a Video-SafetyBench sample.

    Returns 1 (unsafe) for all harmful subcategories/categories,
    0 (safe) for benign content.
    """
    # Any non-empty subcategory or category from the harmful split is unsafe.
    # Callers already set 0 for benign/videochatgpt samples,
    # so this is only called for harmful samples.
    return 1
