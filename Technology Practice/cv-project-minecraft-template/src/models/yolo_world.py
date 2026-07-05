"""
YOLO-World модель для детекции мобов Minecraft
(Zero-shot детекция с текстовыми промптами)
"""
from datetime import datetime
import pandas as pd
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import Dict, List, Union, Optional
import os
from typing import List
from .base_detector import BaseDetector
from ultralytics import YOLO


class YOLOWorldDetector(BaseDetector):
    """Детектор на основе YOLO-World"""

    def __init__(self, config: dict, class_names: List[str]):
        super().__init__(config, class_names)
        self.variant = config.get('variant', 'yolov8s-world')
        self.pretrained = config.get('pretrained', True)
        self.conf_threshold = config.get('conf_threshold', 0.25)
        self.prompt = config.get('prompt', 'creeper. skeleton. spider. zombie. enderman.')

    def build_model(self):
        """Создание модели YOLO-World"""
        # Правильные названия: 'yolov8s-world.pt', 'yolov8m-world.pt', 'yolov8l-world.pt'
        model_name = self.variant
        self.model = YOLO(model_name)  # <-- ИСПРАВЛЕНО!

        # ВАЖНО: устанавливаем классы для open-vocabulary детекции
        self.model.set_classes(self.class_names)

        print(f"[YOLO-World] Model {model_name} built with {len(self.class_names)} classes")

    def train(self, train_loader: DataLoader, val_loader: DataLoader, data_yaml_path: str) -> Dict:
        """Обучение YOLO-World"""
        if self.model is None:
            self.build_model()
        epochs = self.config.get('epochs', 80)
        image_size = self.config.get('image_size', 640)
        batch = self.config.get('batch_size', 16)
        lr = self.config.get('lr', 0.001)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        clean_variant = self.variant.replace('.pt', '')
        save_dir = os.path.join(project_root, 'results', f'yolo_world_{clean_variant}_{timestamp}')
        results = self.model.train(
            data=data_yaml_path,
            epochs=epochs,
            imgsz=image_size,
            batch=batch,
            lr0=lr,
            device=self.device,
            project=save_dir,
            name='train',
            exist_ok=True,
            verbose=True,
            save_period=10
        )
        history = {}
        csv_path = Path(results.save_dir) / 'results.csv'
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            for col in df.columns:
                history[col] = df[col].tolist()
        else:
            # Fallback: только финальные значения
            history = {
                'metrics/mAP50(B)': [results.results_dict.get('metrics/mAP50(B)', 0)],
                'metrics/mAP50-95(B)': [results.results_dict.get('metrics/mAP50-95(B)', 0)],
                'metrics/precision(B)': [results.results_dict.get('metrics/precision(B)', 0)],
                'metrics/recall(B)': [results.results_dict.get('metrics/recall(B)', 0)]
            }

        return {
            'metrics': results.results_dict,
            'history': history,
            'model_path': str(results.save_dir)
        }

    def predict(self, image: Union[np.ndarray, torch.Tensor]) -> Dict:
        """Предсказание для одного изображения"""
        if self.model is None:
            raise ValueError("Model not built or loaded")

        if isinstance(image, torch.Tensor):
            image = image.permute(1, 2, 0).cpu().numpy()
            image = (image * 255).astype(np.uint8)

        results = self.model(image, conf=self.conf_threshold)[0]

        if len(results.boxes) > 0:
            boxes = results.boxes.xyxy.cpu().numpy()
            scores = results.boxes.conf.cpu().numpy()
            labels = results.boxes.cls.cpu().numpy().astype(int)
        else:
            boxes = []
            scores = []
            labels = []

        return {
            'boxes': boxes.tolist() if len(boxes) > 0 else [],
            'scores': scores.tolist() if len(scores) > 0 else [],
            'labels': labels.tolist() if len(labels) > 0 else [],
            'class_names': [self.class_names[l] for l in labels if 0 <= l < self.num_classes]
        }

    def predict_batch(self, images: List[Union[np.ndarray, torch.Tensor]]) -> List[Dict]:
        return [self.predict(img) for img in images]

    def save(self, path: str) -> None:
        if self.model is not None:
            self.model.save(path)
            print(f"[YOLO-World] Model saved to {path}")

    def load(self, path: str) -> None:
        if os.path.exists(path):
            self.model = YOLO(path)
            print(f"[YOLO-World] Model loaded from {path}")

    def get_model_info(self) -> Dict:
        return {
            'name': 'YOLO-World',
            'variant': self.variant,
            'parameters': self.count_parameters(),
            'device': str(self.device),
            'prompt': self.prompt
        }

def create_yolo_world_model(
    variant: str = 'yolov8s-world',
    class_names: List[str] = None,
    device: str = 'cpu'
):
    """
    Создаёт и настраивает модель YOLO-World.

    Args:
        variant: Название модели ('yolov8s-world.pt', 'yolov8m-world.pt', 'yolov8l-world.pt')
        class_names: Список имён классов для промптов
        device: Устройство ('cpu' или 'cuda')

    Returns:
        YOLO: Настроенная модель
    """
    model = YOLO(variant)

    if class_names:
        model.set_classes(class_names)

    if device != 'cpu':
        model.to(device)

    return model