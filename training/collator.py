"""Compatibility exports for training collators."""

from src.common.collator import SafeGemVideoCollator, SafeLLaVAVideoCollator

VideoSafetyCollator = SafeGemVideoCollator

__all__ = ["SafeGemVideoCollator", "SafeLLaVAVideoCollator", "VideoSafetyCollator"]
