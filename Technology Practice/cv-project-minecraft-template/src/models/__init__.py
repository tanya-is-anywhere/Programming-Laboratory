"""
Модуль с моделями детекции объектов
"""

from .base_detector import BaseDetector
from .yolo import YOLODetector
from .faster_rcnn import FasterRCNNDetector
from .detr import DETRDetector
from .rt_detr import create_rt_detr_model
from .yolo_world import YOLOWorldDetector

__all__ = [
    'BaseDetector',
    'YOLODetector',
    'FasterRCNNDetector',
    'DETRDetector',
    'YOLOWorldDetector'
]

# Реестр моделей для фабрики
MODEL_REGISTRY = {
    'yolo': YOLODetector,
    'faster_rcnn': FasterRCNNDetector,
    'detr': DETRDetector,
    'yolo_world': YOLOWorldDetector
}

def get_model(model_name: str, config: dict = None, class_names: list = None):
    """
    Фабрика для создания моделей

    Args:
        model_name: Название модели (yolo, faster_rcnn, detr, rt_detr, yolo_world)
        config: Конфигурация модели
        class_names: Список имен классов

    Returns:
        Инстанс модели детекции
    """
    if model_name not in MODEL_REGISTRY:
        available = list(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model '{model_name}'. Available: {available}")

    if config is None:
        config = {}

    if class_names is None:
        class_names = ["creeper", "skeleton", "spider", "zombie", "enderman"]

    return MODEL_REGISTRY[model_name](config, class_names)
