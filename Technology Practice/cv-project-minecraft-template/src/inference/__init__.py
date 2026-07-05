"""
Модуль для инференса на видео и в реальном времени
"""

from .video_processor import VideoProcessor, process_video
from .realtime_detector import RealtimeDetector

__all__ = [
    'VideoProcessor',
    'process_video',
    'RealtimeDetector'
]
