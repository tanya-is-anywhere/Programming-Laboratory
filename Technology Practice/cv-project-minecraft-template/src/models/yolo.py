from datetime import datetime
import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import Dict, List, Union, Optional
import os
from .base_detector import BaseDetector
from ultralytics import YOLO
import pandas as pd
from pathlib import Path

class YOLODetector(BaseDetector):
    """Детектор на основе YOLO (поддерживает YOLOv8, YOLO-World, YOLOv9, YOLOv10)"""
    def __init__(self, config: dict, class_names: List[str]):
        super().__init__(config, class_names)
        self.variant = config.get('variant', 'yolov8n')
        self.pretrained = config.get('pretrained', True)
        self.image_size = config.get('image_size', 640)
        self.conf_threshold = config.get('conf_threshold', 0.25)
        self.model = None

    def build_model(self) -> None:
        """Создание модели YOLO"""
        # Определяем название модели
        if self.pretrained:
            model_name = f"{self.variant}"
        else:
            model_name = self.variant

        # Проверяем, есть ли локальный файл
        if os.path.exists(model_name):
            self.model = YOLO(model_name)
        else:
            self.model = YOLO(self.variant)

        # Переносим на устройство
        if self.device != 'cpu':
            self.model.to(self.device)


    def train(self, train_loader: DataLoader, val_loader: DataLoader, data_yaml_path: str) -> Dict:
        """Обучение YOLO"""
        if self.model is None:
            self.build_model()
        epochs = self.config.get('epochs', 100)
        batch = self.config.get('batch_size', 16)
        lr = self.config.get('lr', 0.001)
        momentum = self.config.get('momentum', 0.937)
        weight_decay = self.config.get('weight_decay', 0.0005)
        warmup_epochs = self.config.get('warmup_epochs', 3)
        box_loss_gain = self.config.get('box', 7.5)
        cls_loss_gain = self.config.get('cls', 0.5)
        dfl_loss_gain = self.config.get('dfl', 1.5)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        save_dir = os.path.join(project_root, 'results', f'yolo_{self.variant}_{timestamp}')
        # Запускаем обучение
        results = self.model.train(
            data=data_yaml_path,
            epochs=epochs,
            imgsz=self.image_size,
            batch=batch,
            lr0=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            warmup_epochs=warmup_epochs,
            box=box_loss_gain,
            cls=cls_loss_gain,
            dfl=dfl_loss_gain,
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
            raise ValueError("Model not built or loaded. Call build_model() first.")

        # Если это тензор, конвертируем в numpy
        if isinstance(image, torch.Tensor):
            image = image.permute(1, 2, 0).cpu().numpy()
            image = (image * 255).astype(np.uint8)

        # Запускаем инференс
        results = self.model(image, conf=self.conf_threshold)[0]

        # Извлекаем результаты
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
            'labels': labels.tolist() if len(labels) > 0 else [],
            'class_names': [self.class_names[l] for l in labels] if len(labels) > 0 else []
        }

    def predict_batch(self, images: List[Union[np.ndarray, torch.Tensor]]) -> List[Dict]:
        """Предсказание для батча"""
        return [self.predict(img) for img in images]

    def save(self, path: str) -> None:
        """Сохранение модели"""
        if self.model is not None:
            self.model.save(path)
            print(f"[YOLO] Model saved to {path}")
        else:
            raise ValueError("No model to save")

    def load(self, path: str) -> None:
        """Загрузка модели"""
        self.model = YOLO(path)
        print(f"[YOLO] Model loaded from {path}")

    def get_model_info(self) -> Dict:
        """Информация о модели"""
        return {
            'name': f'YOLO_{self.variant}',
            'variant': self.variant,
            'parameters': self.count_parameters(),
            'device': str(self.device),
            'pretrained': self.pretrained,
            'image_size': self.image_size
        }

    def validate(self, data_yaml_path: str) -> Dict:
        """
        Валидация модели на тестовой/валидационной выборке

        Args:
            data_yaml_path: Путь к data.yaml файлу

        Returns:
            Dict с метриками:
                - 'mAP_0.5': mAP при IoU=0.5
                - 'mAP_0.5_0.95': средний mAP по 10 порогам IoU
                - 'precision': точность
                - 'recall': полнота
                - 'f1': F1-мера
                - 'per_class': метрики по каждому классу (опционально)
        """
        if self.model is None:
            raise ValueError("Model not loaded or built. Call build_model() or load() first.")
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        save_dir = os.path.join(project_root, 'results', 'logs', f'{self.variant}_val')
        # Запускаем валидацию
        results = self.model.val(
            data=data_yaml_path,
            imgsz=self.image_size,
            batch=self.config.get('batch_size', 16),
            device=self.device,
            project=save_dir,
            name='',
            exist_ok=True,
            verbose=True
        )

        # Извлекаем метрики
        metrics = {
            'mAP_0.5': results.box.map50,
            'mAP_0.5_0.95': results.box.map,
            'precision': results.box.mp,
            'recall': results.box.mr,
            'f1': results.box.f1,
            'speed': results.speed,  # скорость инференса
        }

        # Добавляем метрики по классам (опционально)
        if hasattr(results.box, 'ap_class_index'):
            per_class = {}
            for i, cls_idx in enumerate(results.box.ap_class_index):
                class_name = self.class_names[cls_idx] if cls_idx < len(self.class_names) else f'class_{cls_idx}'
                per_class[class_name] = {
                    'precision': results.box.p[i] if i < len(results.box.p) else 0,
                    'recall': results.box.r[i] if i < len(results.box.r) else 0,
                    'mAP_0.5': results.box.ap50[i] if i < len(results.box.ap50) else 0,
                    'mAP_0.5_0.95': results.box.ap[i] if i < len(results.box.ap) else 0
                }
            metrics['per_class'] = per_class

        print(f"[YOLO] Validation completed: mAP_0.5 = {metrics['mAP_0.5']:.4f}")
        return metrics
