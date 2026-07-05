"""
rt_detr.py - RT-DETR модель через Ultralytics (как YOLO)
"""

from ultralytics import RTDETR
import torch
import numpy as np
from typing import Dict, List, Union, Optional
import os


class RTDETRModel:
    """
    Обёртка для RT-DETR через Ultralytics.
    Работает как YOLO, но с RT-DETR архитектурой.
    """

    def __init__(self, num_classes: int = 5):
        self.num_classes = num_classes
        self.model = None
        self.device = 'cpu'
        self.variant = 'rtdetr-l.pt'

    def build_model(self, variant: str = 'rtdetr-l.pt') -> None:
        """Создание модели RT-DETR"""
        self.variant = variant
        self.model = RTDETR(variant)
        print(f"[RT-DETR] Model built")

    def count_parameters(self) -> int:
        """Подсчёт количества параметров"""
        if self.model is None:
            return 0
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'parameters'):
            return sum(p.numel() for p in self.model.model.parameters() if p.requires_grad)
        return 0

    def train(
        self,
        data_yaml_path: str,
        epochs: int = 50,
        imgsz: int = 320,
        batch: int = 16,
        lr0: float = 0.001,
        device: str = 'cpu',
        project: str = 'results/logs',
        name: str = 'rtdetr',
        exist_ok: bool = True,
        verbose: bool = True
    ) -> Dict:
        """Обучение RT-DETR через Ultralytics"""
        if self.model is None:
            self.build_model()

        results = self.model.train(
            data=data_yaml_path,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            lr0=lr0,
            device=device,
            project=project,
            name=name,
            exist_ok=exist_ok,
            verbose=verbose
        )

        return {
            'model_path': str(results.save_dir),
            'results': results
        }

    def predict(self, image: Union[str, np.ndarray], conf: float = 0.25) -> Dict:
        """Предсказание для одного изображения"""
        if self.model is None:
            raise ValueError("Model not built or loaded")

        results = self.model.predict(image, conf=conf)[0]

        if len(results.boxes) > 0:
            boxes = results.boxes.xyxy.cpu().numpy()
            scores = results.boxes.conf.cpu().numpy()
            labels = results.boxes.cls.cpu().numpy().astype(int)
        else:
            boxes = np.array([])
            scores = np.array([])
            labels = np.array([])

        return {
            'boxes': boxes.tolist() if len(boxes) > 0 else [],
            'scores': scores.tolist() if len(scores) > 0 else [],
            'labels': labels.tolist() if len(labels) > 0 else []
        }

    def predict_batch(self, images: List[Union[str, np.ndarray]]) -> List[Dict]:
        """Предсказание для батча"""
        return [self.predict(img) for img in images]

    def save(self, path: str) -> None:
        """Сохранение модели"""
        if self.model is not None:
            self.model.save(path)
            print(f"[RT-DETR] Model saved to {path}")

    def load(self, path: str) -> None:
        """Загрузка модели"""
        self.model = RTDETR(path)
        print(f"[RT-DETR] Model loaded from {path}")

    def get_model_info(self) -> Dict:
        """Информация о модели"""
        return {
            'name': 'RT-DETR',
            'variant': self.variant,
            'parameters': self.count_parameters(),
            'device': str(self.device)
        }


def create_rt_detr_model(config: dict):
    """Быстрое создание RT-DETR модели"""
    num_classes = config.get('num_classes', 5)
    variant = config.get('variant', 'rtdetr-l.pt')
    model = RTDETRModel(num_classes)
    model.build_model(variant)
    return model