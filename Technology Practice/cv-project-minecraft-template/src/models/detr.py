import torch
from transformers import DetrForObjectDetection, DetrImageProcessor
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Union, Optional
import os
from transformers import DetrConfig
from .base_detector import BaseDetector
from transformers import DetrForObjectDetection

class DETRDetector(BaseDetector):
    """Детектор на основе DETR (Facebook Detection Transformer)"""

    def __init__(self, config: dict, class_names: List[str]):
        super().__init__(config, class_names)
        self.backbone = config.get('backbone', 'resnet50')
        self.pretrained = config.get('pretrained', True)
        self.image_size = config.get('image_size', 640)
        self.conf_threshold = config.get('conf_threshold', 0.5)
        self.num_queries = config.get('num_queries', 100)
        self.hidden_dim = config.get('hidden_dim', 256)
        self.model = None
        self.processor = None

    def build_model(self) -> None:
        """Создание модели DETR"""
        # Используем стандартную модель из transformers
        model_name = "facebook/detr-resnet-50"

        # Инициализируем процессор
        self.processor = DetrImageProcessor.from_pretrained(model_name)

        # Создаём модель с нужным числом классов
        if self.pretrained:
            self.model = DetrForObjectDetection.from_pretrained(
                model_name,
                num_labels=self.num_classes,
                ignore_mismatched_sizes=True
            )
        else:
            config = DetrConfig(
                num_labels=self.num_classes,
                hidden_dim=self.hidden_dim,
                num_queries=self.num_queries,
                backbone=self.backbone
            )
            self.model = DetrForObjectDetection(config)

        self.model.to(self.device)
        print(f"[DETR] Model built with {self.count_parameters():,} parameters")
        print(f"[DETR] Number of classes: {self.num_classes}")

    def train(self, train_loader: DataLoader, val_loader: DataLoader, **kwargs) -> Dict:
        """Обучение DETR"""
        if self.model is None:
            self.build_model()

        epochs = self.config.get('epochs', 150)
        lr = self.config.get('lr', 0.0001)
        lr_backbone = self.config.get('lr_backbone', 0.00001)
        weight_decay = self.config.get('weight_decay', 0.0001)

        # Оптимизатор с разными lr для бэкбона и остальной части
        param_dicts = [
            {"params": [p for n, p in self.model.named_parameters() if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in self.model.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": lr_backbone,
            },
        ]
        optimizer = torch.optim.AdamW(param_dicts, lr=lr, weight_decay=weight_decay)

        # Scheduler (cosine decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        self.model.train()
        history = {'train_loss': [], 'val_loss': []}

        for epoch in range(epochs):
            epoch_loss = 0.0
            self.model.train()

            for batch in train_loader:
                pixel_values = batch['pixel_values'].to(self.device)
                labels = batch['labels']

                # Подготовка targets для DETR
                target_list = []
                for i in range(len(pixel_values)):
                    boxes = labels['boxes'][i].to(self.device)
                    class_labels = labels['labels'][i].to(self.device)

                    # DETR ожидает формат: {"class_labels": ..., "boxes": ...}
                    if len(boxes) > 0:
                        target_list.append({
                            "class_labels": class_labels,
                            "boxes": boxes
                        })
                    else:
                        target_list.append({
                            "class_labels": torch.zeros((0,), dtype=torch.long, device=self.device),
                            "boxes": torch.zeros((0, 4), device=self.device)
                        })

                # Forward
                outputs = self.model(pixel_values=pixel_values, labels=target_list)
                loss = outputs.loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            scheduler.step()

            avg_loss = epoch_loss / len(train_loader)
            history['train_loss'].append(avg_loss)

            # Валидация
            val_loss = self.validate(val_loader)
            history['val_loss'].append(val_loss['val_loss'])

            print(f"[DETR] Epoch {epoch+1}/{epochs}, Train Loss: {avg_loss:.4f}, Val Loss: {val_loss['val_loss']:.4f}")

        # Сохраняем модель
        save_dir = 'results/logs/detr'
        os.makedirs(save_dir, exist_ok=True)
        self.model.save_pretrained(save_dir)

        return {
            'metrics': {'train_loss': avg_loss, 'val_loss': val_loss['val_loss']},
            'model_path': save_dir,
            'history': history
        }

    def predict(self, image: Union[np.ndarray, torch.Tensor]) -> Dict:
        """Предсказание для одного изображения"""
        if self.model is None or self.processor is None:
            raise ValueError("Model or processor not built")

        self.model.eval()

        if isinstance(image, torch.Tensor):
            image = image.permute(1, 2, 0).cpu().numpy()
            image = (image * 255).astype(np.uint8)

        # DETR использует свой процессор
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Постобработка
        target_sizes = torch.tensor([image.shape[:2]]).to(self.device)
        results = self.processor.post_process_object_detection(
            outputs,
            target_sizes=target_sizes,
            threshold=self.conf_threshold
        )[0]

        boxes = results['boxes'].cpu().numpy()
        scores = results['scores'].cpu().numpy()
        labels = results['labels'].cpu().numpy().astype(int)

        return {
            'boxes': boxes.tolist() if len(boxes) > 0 else [],
            'scores': scores.tolist() if len(scores) > 0 else [],
            'labels': labels.tolist() if len(labels) > 0 else [],
            'class_names': [self.class_names[l] for l in labels] if len(labels) > 0 else []
        }

    def predict_batch(self, images: List[Union[np.ndarray, torch.Tensor]]) -> List[Dict]:
        return [self.predict(img) for img in images]

    def save(self, path: str) -> None:
        if self.model is not None:
            self.model.save_pretrained(path)
            print(f"[DETR] Model saved to {path}")

    def load(self, path: str) -> None:
        if self.model is None:
            self.build_model()
        self.model = DetrForObjectDetection.from_pretrained(path)
        self.model.to(self.device)
        print(f"[DETR] Model loaded from {path}")

    def get_model_info(self) -> Dict:
        return {
            'name': 'DETR',
            'backbone': self.backbone,
            'parameters': self.count_parameters(),
            'device': str(self.device),
            'pretrained': self.pretrained,
            'num_queries': self.num_queries,
            'image_size': self.image_size
        }

    def validate(self, val_loader: DataLoader) -> Dict:
        """Валидация модели"""
        if self.model is None:
            raise ValueError("Model not built or loaded")

        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                pixel_values = batch['pixel_values'].to(self.device)
                labels = batch['labels']

                target_list = []
                for i in range(len(pixel_values)):
                    boxes = labels['boxes'][i].to(self.device)
                    class_labels = labels['labels'][i].to(self.device)
                    if len(boxes) > 0:
                        target_list.append({
                            "class_labels": class_labels,
                            "boxes": boxes
                        })
                    else:
                        target_list.append({
                            "class_labels": torch.zeros((0,), dtype=torch.long, device=self.device),
                            "boxes": torch.zeros((0, 4), device=self.device)
                        })

                outputs = self.model(pixel_values=pixel_values, labels=target_list)
                loss = outputs.loss
                total_loss += loss.item()

        avg_loss = total_loss / len(val_loader)
        return {'val_loss': avg_loss}

def create_detr_model(
    num_classes: int = 5,
    pretrained: str = 'facebook/detr-resnet-50',
    device: str = 'cpu'
) -> DetrForObjectDetection:
    """
    Создаёт модель DETR с указанным числом классов.

    Args:
        num_classes: Количество классов (без фона)
        pretrained: Имя предобученной модели
        device: Устройство ('cpu' или 'cuda')

    Returns:
        DetrForObjectDetection: Модель DETR
    """
    model = DetrForObjectDetection.from_pretrained(
        pretrained,
        num_labels=num_classes + 1,
        ignore_mismatched_sizes=True
    ).to(device)

    return model

