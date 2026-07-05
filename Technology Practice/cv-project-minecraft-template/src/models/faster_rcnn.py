import torch
import torch.nn as nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn, fasterrcnn_mobilenet_v3_large_fpn
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Union, Optional
import os
import torchvision.transforms as T
from .base_detector import BaseDetector


class FasterRCNNDetector(BaseDetector):
    """Детектор на основе Faster R-CNN"""

    def __init__(self, config: dict, class_names: List[str]):
        super().__init__(config, class_names)
        self.backbone = config.get('backbone', 'resnet50')
        self.pretrained = config.get('pretrained', True)
        self.image_size = config.get('image_size', 640)
        self.conf_threshold = config.get('conf_threshold', 0.5)
        self.model = None

    def build_model(self) -> None:
        """Создание модели Faster R-CNN"""
        # Создаём модель напрямую из torchvision
        self.model = fasterrcnn_mobilenet_v3_large_fpn(weights='DEFAULT')

        # Заменяем голову классификатора
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes)

        self.model.to(self.device)
        print(f"[Faster R-CNN] Model built with {self.count_parameters():,} parameters")
        print(f"[Faster R-CNN] Number of classes: {self.num_classes}")

    def train(self, train_loader: DataLoader, val_loader: DataLoader, **kwargs) -> Dict:
        """Обучение Faster R-CNN"""
        if self.model is None:
            self.build_model()
        
        epochs = self.config.get('epochs', 100)
        lr = self.config.get('lr', 0.005)
        momentum = self.config.get('momentum', 0.9)
        weight_decay = self.config.get('weight_decay', 0.0005)
        
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
        
        # Scheduler
        step_size = self.config.get('lr_step_size', 30)
        gamma = self.config.get('lr_gamma', 0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        
        self.model.train()
        
        # История для логирования
        history = {'train_loss': [], 'val_loss': []}
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            self.model.train()
            
            for batch in train_loader:
                images = batch['pixel_values'].to(self.device)
                targets = batch['labels']
                
                # Подготовка targets для Faster R-CNN
                target_list = []
                for i in range(len(images)):
                    boxes = targets['boxes'][i].to(self.device)
                    labels = targets['labels'][i].to(self.device)
                    # Проверяем, что есть объекты
                    if len(boxes) > 0:
                        target_list.append({'boxes': boxes, 'labels': labels})
                    else:
                        # Пустые изображения (background)
                        target_list.append({'boxes': torch.zeros((0, 4), device=self.device),
                                            'labels': torch.zeros((0,), dtype=torch.int64, device=self.device)})
                
                loss_dict = self.model(images, target_list)
                losses = sum(loss for loss in loss_dict.values())
                
                optimizer.zero_grad()
                losses.backward()
                optimizer.step()
                
                epoch_loss += losses.item()
            
            scheduler.step()
            
            avg_loss = epoch_loss / len(train_loader)
            history['train_loss'].append(avg_loss)
            
            # Валидация после каждой эпохи
            val_loss = self.validate(val_loader)
            history['val_loss'].append(val_loss['val_loss'])
            
            print(f"[Faster R-CNN] Epoch {epoch+1}/{epochs}, Train Loss: {avg_loss:.4f}, Val Loss: {val_loss['val_loss']:.4f}")
        
        # Сохраняем модель
        save_dir = 'results/logs/faster_rcnn'
        os.makedirs(save_dir, exist_ok=True)
        model_path = f'{save_dir}/best.pt'
        torch.save(self.model.state_dict(), model_path)
        
        return {
            'metrics': {'train_loss': avg_loss, 'val_loss': val_loss['val_loss']},
            'model_path': model_path,
            'history': history
        }

    def predict(self, image: Union[np.ndarray, torch.Tensor]) -> Dict:
        """Предсказание для одного изображения"""
        if self.model is None:
            raise ValueError("Model not built or loaded")
        
        self.model.eval()
        
        if isinstance(image, torch.Tensor):
            image = image.permute(1, 2, 0).cpu().numpy()
            image = (image * 255).astype(np.uint8)
        
        # Трансформации
        transform = T.Compose([
            T.ToTensor(),
            T.Resize((self.image_size, self.image_size)),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        image_tensor = transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(image_tensor)[0]
        
        boxes = predictions['boxes'].cpu().numpy()
        scores = predictions['scores'].cpu().numpy()
        labels = predictions['labels'].cpu().numpy().astype(int)
        
        # Фильтруем по порогу
        mask = scores >= self.conf_threshold
        boxes = boxes[mask].tolist()
        scores = scores[mask].tolist()
        labels = labels[mask].tolist()
        
        return {
            'boxes': boxes,
            'scores': scores,
            'labels': labels,
            'class_names': [self.class_names[l] for l in labels] if labels else []
        }

    def predict_batch(self, images: List[Union[np.ndarray, torch.Tensor]]) -> List[Dict]:
        return [self.predict(img) for img in images]

    def save(self, path: str) -> None:
        if self.model is not None:
            torch.save(self.model.state_dict(), path)
            print(f"[Faster R-CNN] Model saved to {path}")

    def load(self, path: str) -> None:
        if self.model is None:
            self.build_model()
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.to(self.device)
        print(f"[Faster R-CNN] Model loaded from {path}")

    def get_model_info(self) -> Dict:
        return {
            'name': 'Faster R-CNN',
            'backbone': self.backbone,
            'parameters': self.count_parameters(),
            'device': str(self.device),
            'pretrained': self.pretrained,
            'image_size': self.image_size
        }

    def validate(self, val_loader: DataLoader) -> Dict:
        """Валидация модели на валидационной выборке"""
        if self.model is None:
            raise ValueError("Model not built or loaded")
        
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch['pixel_values'].to(self.device)
                targets = batch['labels']
                
                target_list = []
                for i in range(len(images)):
                    boxes = targets['boxes'][i].to(self.device)
                    labels = targets['labels'][i].to(self.device)
                    if len(boxes) > 0:
                        target_list.append({'boxes': boxes, 'labels': labels})
                    else:
                        target_list.append({'boxes': torch.zeros((0, 4), device=self.device),
                                            'labels': torch.zeros((0,), dtype=torch.int64, device=self.device)})
                
                loss_dict = self.model(images, target_list)
                losses = sum(loss for loss in loss_dict.values())
                total_loss += losses.item()
        
        avg_loss = total_loss / len(val_loader)
        return {'val_loss': avg_loss}
