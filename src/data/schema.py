"""Shared schema helpers for processed training and evaluation samples."""

from typing import TypedDict


class VideoSafetySample(TypedDict, total=False):
    dataset: str
    split: str
    question_id: str
    video_path: str
    question: str
    answer: str
    safety_label: int
    category: str
    subcategory: str

