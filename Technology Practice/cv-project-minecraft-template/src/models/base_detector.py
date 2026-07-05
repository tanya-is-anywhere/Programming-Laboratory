from abc import ABC, abstractmethod
from typing import Dict, List, Union, Optional
import numpy as np
import torch
from torch.utils.data import DataLoader


class BaseDetector(ABC):
    """
    Базовый класс для всех моделей детекции.
    Все модели должны наследоваться от этого класса.
    """

    def __init__(self, config: dict, class_names: List[str]):
        self.config = config
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.device = config.get('device', 'cpu')
        self.model = None

    @abstractmethod
    def build_model(self) -> None:
        """Создание и инициализация модели"""
        pass

    @abstractmethod
    def train(self, train_loader: DataLoader, val_loader: DataLoader, **kwargs) -> Dict:
        """
        Обучение модели

        Returns:
            Dict с метриками и путём к сохранённой модели
        """
        pass

    @abstractmethod
    def predict(self, image: Union[np.ndarray, torch.Tensor]) -> Dict:
        """
        Предсказание для одного изображения

        Returns:
            Dict с ключами:
                - 'boxes': List[List[float]] [x1, y1, x2, y2]
                - 'scores': List[float]
                - 'labels': List[int]
                - 'class_names': List[str]
        """
        pass

    @abstractmethod
    def predict_batch(self, images: List[Union[np.ndarray, torch.Tensor]]) -> List[Dict]:
        """Предсказание для батча"""
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """Сохранение весов модели"""
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """Загрузка весов модели"""
        pass

    @abstractmethod
    def get_model_info(self) -> Dict:
        """Информация о модели: параметры, версия, размер"""
        pass

    def count_parameters(self) -> int:
        """Подсчёт количества параметров"""
        if self.model is None:
            return 0
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def to(self, device: str):
        """Перенос модели на устройство"""
        self.device = device
        if self.model is not None:
            if hasattr(self.model, 'to'):
                self.model.to(device)
        return self

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(classes={self.num_classes}, device={self.device})"
