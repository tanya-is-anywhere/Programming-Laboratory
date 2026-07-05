from typing import Dict, List, Optional
from .base_detector import BaseDetector
from .yolo import YOLODetector
from .yolo_world import YOLOWorldDetector
from .detr import DETRDetector
from .rt_detr import create_rt_detr_model


def create_detector(model_name: str, config: dict, class_names: List[str]) -> BaseDetector:
    """
    Фабрика для создания детектора по имени модели.

    Args:
        model_name: 'yolo', 'yolo_world', 'faster_rcnn', 'detr', 'rt_detr'
        config: Словарь с гиперпараметрами
        class_names: Список имён классов

    Returns:
        Экземпляр BaseDetector
    """
    model_name = model_name.lower()

    if model_name in ['yolo', 'yolov8', 'yolov9', 'yolov10']:
        return YOLODetector(config, class_names)

    elif model_name in ['yolo_world', 'yoloworld']:
        # Для YOLO-World используем тот же класс, но с другим вариантом
        return YOLOWorldDetector(config, class_names)

    elif model_name == 'faster_rcnn':
        from .faster_rcnn import FasterRCNNDetector
        return FasterRCNNDetector(config, class_names)

    elif model_name == 'detr':
        from .detr import DETRDetector
        return DETRDetector(config, class_names)


    elif model_name in ['rtdetr', 'rt_detr']:
        return create_rt_detr_model(config)

    else:
        raise ValueError(
            f"Unknown model name: {model_name}. Available: 'yolo', 'yolo_world', 'faster_rcnn', 'detr', 'rt_detr'")
