"""Compatibility exports for backend-specific dataset implementations."""

from src.data.video_utils import resize_frame as _resize_frame
from src.data.video_utils import sample_frames_from_video, select_representative_frame
from src.models.safegem.dataset import VideoSafetyDataset as VideoSafetyDataset
from src.models.safegem.dataset import VideoSafetyDataset as VideoSafetyDatasetSafeGem
from src.models.safeqwen.dataset import VideoSafetyDataset as VideoSafetyDatasetSafeQwen
from src.models.safellava.dataset import VideoSafetyDataset as VideoSafetyDatasetLLaVA

__all__ = [
    "_resize_frame",
    "sample_frames_from_video",
    "select_representative_frame",
    "VideoSafetyDataset",
    "VideoSafetyDatasetSafeGem",
    "VideoSafetyDatasetSafeQwen",
    "VideoSafetyDatasetLLaVA",
]
