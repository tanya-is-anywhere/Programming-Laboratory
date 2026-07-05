"""
Модуль для загрузки и предобработки датасета Minecraft Mobs.
Поддерживает форматы: YOLO, COCO, Pascal VOC.
"""

import os
import json
import random
from collections import defaultdict
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Callable
import logging
from torchvision import transforms as T
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import yaml

logger = logging.getLogger(__name__)


class MinecraftMobsDataset(Dataset):
    """
    Датасет для детекции мобов в Minecraft.

    Поддерживает аннотации в формате YOLO (нормализованные координаты)
    и может конвертировать их в другие форматы (COCO, Pascal VOC).
    """

    def __init__(
            self,
            data_dir: str,
            split: str = 'train',
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            img_size: int = 640,
            num_classes: int = 5,
            class_names: List[str] = None,
            box_format: str = 'xyxy',
            debug: bool = False
    ):
        """
        Args:
            data_dir: Путь к директории с данными
            split: 'train', 'val', 'test'
            transform: Альбументации для изображений
            target_transform: Трансформации для аннотаций
            img_size: Размер изображения после ресайза
            num_classes: Количество классов
            class_names: Имена классов
            debug: Режим отладки (меньше логов, больше проверок)
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.img_size = img_size
        self.num_classes = num_classes
        self.class_names = class_names or [f"class_{i}" for i in range(num_classes)]
        self.box_format = box_format
        self.debug = debug

        # Пути к данным
        self.images_dir = self.data_dir / split / 'images'
        self.labels_dir = self.data_dir / split / 'labels'

        # Проверка существования директорий
        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")

        # Загружаем список изображений
        self.image_files = self._load_image_files()

        if len(self.image_files) == 0:
            raise RuntimeError(f"No images found in {self.images_dir}")

        logger.info(f"Loaded {len(self.image_files)} images for {split} split")

        # Трансформации
        self.transform = transform or self._get_default_transforms()
        self.target_transform = target_transform

        # Кэш для аннотаций
        self._annotations_cache = {}

        # Статистика датасета
        self._dataset_stats = None

    def _load_image_files(self) -> List[str]:
        """Загрузка списка изображений с поддержкой разных форматов"""
        supported_formats = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'}
        image_files = []

        for ext in supported_formats:
            image_files.extend(self.images_dir.glob(f'*{ext}'))
            image_files.extend(self.images_dir.glob(f'*{ext.upper()}'))

        # Сортируем для воспроизводимости
        image_files = sorted([f.name for f in image_files])

        # Проверяем наличие аннотаций
        valid_files = []
        for img_file in image_files:
            label_file = self._get_label_path(img_file)
            if self.split == 'test' or label_file.exists():
                valid_files.append(img_file)
            elif self.debug:
                logger.warning(f"No annotation found for {img_file}, skipping...")

        return valid_files

    def _get_label_path(self, img_file: str) -> Path:
        """Получение пути к файлу аннотации"""
        name = Path(img_file).stem
        return self.labels_dir / f"{name}.txt"

    def _get_torchvision_transforms(self) -> T.Compose:
        """Трансформации через Torchvision (без Albumentations)"""
        if self.split == 'train':
            return T.Compose([
                T.ToPILImage(),
                T.Resize((self.img_size, self.img_size)),
                T.RandomHorizontalFlip(p=0.5),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            return T.Compose([
                T.ToPILImage(),
                T.Resize((self.img_size, self.img_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Dict:
        img_file = self.image_files[idx]
        img_path = self.images_dir / img_file
        label_path = self._get_label_path(img_file)

        # Загрузка изображения
        image_raw = cv2.imread(str(img_path))
        if image_raw is None:
            raise RuntimeError(f"Failed to load image: {img_path}")
        image_raw = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = image_raw.shape[:2]

        # Загрузка аннотаций в пикселях на оригинальном изображении
        boxes_xyxy_pixels = []
        labels_list = []

        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cls_id = int(parts[0])
                    cx, cy, w, h = map(float, parts[1:5])

                    # Защита от некорректных значений
                    w = max(1e-6, min(1.0, w))
                    h = max(1e-6, min(1.0, h))
                    cx = max(0.0, min(1.0, cx))
                    cy = max(0.0, min(1.0, cy))

                    # Считаем координаты в пикселях на оригинальном изображении
                    x1_px = (cx - w / 2.0) * orig_w
                    y1_px = (cy - h / 2.0) * orig_h
                    x2_px = (cx + w / 2.0) * orig_w
                    y2_px = (cy + h / 2.0) * orig_h

                    # Обрезаем по границам
                    x1_px = max(0, x1_px)
                    y1_px = max(0, y1_px)
                    x2_px = min(orig_w, x2_px)
                    y2_px = min(orig_h, y2_px)

                    if x2_px <= x1_px or y2_px <= y1_px:
                        continue

                    boxes_xyxy_pixels.append([x1_px, y1_px, x2_px, y2_px])
                    labels_list.append(cls_id)

        # Применяем трансформации через Torchvision
        transform = self._get_torchvision_transforms()
        image = transform(image_raw)

        # Масштабируем боксы под новый размер
        new_size = self.img_size
        scale_w = new_size / orig_w
        scale_h = new_size / orig_h

        boxes_resized = []
        for x1, y1, x2, y2 in boxes_xyxy_pixels:
            boxes_resized.append([
                x1 * scale_w,
                y1 * scale_h,
                x2 * scale_w,
                y2 * scale_h
            ])

        if len(boxes_resized) == 0:
            boxes_tensor = torch.empty((0, 4), dtype=torch.float32)
            labels_tensor = torch.empty((0,), dtype=torch.long)
        else:
            boxes_tensor = torch.tensor(boxes_resized, dtype=torch.float32)
            labels_tensor = torch.tensor(labels_list, dtype=torch.long)

        # Возвращаем в едином формате для всех моделей
        return {
            'pixel_values': image,
            'labels': {
                'boxes': boxes_tensor,
                'labels': labels_tensor
            },
            'orig_size': (orig_h, orig_w),
            'image_path': str(img_path),
            'image_id': idx
        }

    def _load_yolo_annotations(self, label_path: Path):
        boxes, labels = [], []
        if label_path.exists():
            for line in open(label_path).readlines():
                parts = line.strip().split()
                if len(parts) >= 5:
                    coords = [max(0.0, min(1.0, float(x))) for x in parts[1:5]]
                    boxes.append(coords)
                    labels.append(int(parts[0]))
        return boxes, labels

    def _convert_boxes(
            self,
            boxes_yolo: List[List[float]],
            orig_w: int,
            orig_h: int,
            target_w: int,
            target_h: int
    ) -> np.ndarray:
        """
        Конвертирует боксы из YOLO формата в нужный формат.

        Args:
            boxes_yolo: List of [cx, cy, w, h] нормализованные (0..1)
            orig_w, orig_h: Оригинальный размер изображения
            target_w, target_h: Размер после ресайза

        Returns:
            np.ndarray (N, 4) в формате, заданном self.box_format
        """
        if not boxes_yolo:
            return np.array([], dtype=np.float32).reshape(0, 4)

        converted = []
        for box in boxes_yolo:
            cx, cy, w, h = box

            if self.box_format == 'cxcywh':
                # Для DETR: нормализованные cx, cy, w, h (относительно target)
                converted.append([cx, cy, w, h])

            elif self.box_format == 'yolo':
                # Для YOLO: просто возвращаем как есть
                converted.append([cx, cy, w, h])

            elif self.box_format == 'pixel_xyxy_orig':
                # Для Faster R-CNN: пиксели на оригинальном изображении
                x1 = (cx - w / 2) * orig_w
                y1 = (cy - h / 2) * orig_h
                x2 = (cx + w / 2) * orig_w
                y2 = (cy + h / 2) * orig_h
                # Клиппинг по границам
                x1 = max(0, min(orig_w, x1))
                y1 = max(0, min(orig_h, y1))
                x2 = max(0, min(orig_w, x2))
                y2 = max(0, min(orig_h, y2))
                converted.append([x1, y1, x2, y2])

            elif self.box_format == 'xyxy':
                # Для RT-DETR: пиксели на изображении после ресайза
                x1 = (cx - w / 2) * target_w
                y1 = (cy - h / 2) * target_h
                x2 = (cx + w / 2) * target_w
                y2 = (cy + h / 2) * target_h
                x1 = max(0, min(target_w, x1))
                y1 = max(0, min(target_h, y1))
                x2 = max(0, min(target_w, x2))
                y2 = max(0, min(target_h, y2))
                converted.append([x1, y1, x2, y2])
            else:
                raise ValueError(f"Unknown box_format: {self.box_format}")

        return np.array(converted, dtype=np.float32)

    def _get_default_transforms(self) -> Callable:
        if self.split == 'train':
            return A.Compose([
                A.Resize(self.img_size, self.img_size),
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ], bbox_params=A.BboxParams(format='yolo', min_visibility=0.0, label_fields=['class_labels']))
        else:
            return A.Compose([
                A.Resize(self.img_size, self.img_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ], bbox_params=A.BboxParams(format='yolo', min_visibility=0.0, label_fields=['class_labels']))
    def _yolo_to_pixel(
            self,
            boxes: List[List[float]],
            img_w: int,
            img_h: int
    ) -> np.ndarray:
        """
        Конвертация из YOLO формата (нормализованные) в пиксели [x1, y1, x2, y2]

        YOLO: [center_x, center_y, width, height] (нормализованные)
        Pixel: [x1, y1, x2, y2]
        """
        pixel_boxes = []
        for box in boxes:
            cx, cy, w, h = box
            x1 = (cx - w / 2) * img_w
            y1 = (cy - h / 2) * img_h
            x2 = (cx + w / 2) * img_w
            y2 = (cy + h / 2) * img_h
            pixel_boxes.append([x1, y1, x2, y2])

        return np.array(pixel_boxes, dtype=np.float32) if pixel_boxes else np.array([])

    def get_dataset_stats(self) -> Dict:
        """Получение статистики датасета"""
        if self._dataset_stats is not None:
            return self._dataset_stats

        stats = {
            'num_images': len(self),
            'num_objects': 0,
            'class_distribution': {name: 0 for name in self.class_names},
            'avg_objects_per_image': 0,
            'images_without_objects': 0
        }

        for idx in range(len(self)):
            img_file = self.image_files[idx]
            label_path = self._get_label_path(img_file)

            if label_path.exists():
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            if 0 <= class_id < self.num_classes:
                                stats['class_distribution'][self.class_names[class_id]] += 1
                                stats['num_objects'] += 1

            if not label_path.exists() or stats['num_objects'] == 0:
                stats['images_without_objects'] += 1

        stats['avg_objects_per_image'] = stats['num_objects'] / max(1, stats['num_images'])

        self._dataset_stats = stats
        return stats

    def create_data_yaml(self, output_path: str) -> str:
        """
        Создание YAML файла для YOLO моделей

        Args:
            output_path: Путь для сохранения

        Returns:
            Путь к созданному файлу
        """
        data_config = {
            'path': str(self.data_dir.absolute()),
            'train': f'{self.split}/images',
            'val': f'{self.split}/images',
            'test': f'{self.split}/images',
            'nc': self.num_classes,
            'names': {i: name for i, name in enumerate(self.class_names)}
        }

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            yaml.dump(data_config, f, default_flow_style=False, allow_unicode=True)

        logger.info(f"Data YAML created at {output_path}")
        return str(output_path)

class MinecraftDatasetRTDETR(Dataset):
    """Датасет для RT-DETR: возвращает боксы в формате xyxy (пиксели)."""

    def __init__(self, data_dir: str, split: str = 'train', img_size: int = 320):
        self.data_dir = Path(data_dir)
        self.split = split
        self.img_size = img_size
        self.images_dir = self.data_dir / split / 'images'
        self.labels_dir = self.data_dir / split / 'labels'

        self.files = sorted([
            f.name for f in self.images_dir.glob('*')
            if f.suffix.lower() in ('.jpg', '.jpeg', '.png')
            and (self.labels_dir / (f.stem + '.txt')).exists()
        ])
        # Для тестов
        self.files = self.files[:16]

        logger.info(f"[{split}] Loaded {len(self.files)} images")

        self.transform = T.Compose([
            T.ToTensor(),
            T.Resize((img_size, img_size)),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_file = self.files[idx]
        img_path = self.images_dir / img_file
        label_path = self.labels_dir / (Path(img_file).stem + '.txt')

        image_raw = cv2.imread(str(img_path))
        image_raw = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = image_raw.shape[:2]

        boxes_xyxy_pixels = []
        labels_list = []

        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cls_id = int(parts[0])
                    cx, cy, w, h = map(float, parts[1:5])

                    cx = max(0.0, min(1.0, cx))
                    cy = max(0.0, min(1.0, cy))
                    w = max(1e-6, min(1.0, w))
                    h = max(1e-6, min(1.0, h))

                    # Считаем координаты в пикселях на оригинальном изображении
                    x1_px = (cx - w / 2.0) * orig_w
                    y1_px = (cy - h / 2.0) * orig_h
                    x2_px = (cx + w / 2.0) * orig_w
                    y2_px = (cy + h / 2.0) * orig_h

                    x1_px = max(0, min(orig_w, x1_px))
                    y1_px = max(0, min(orig_h, y1_px))
                    x2_px = max(0, min(orig_w, x2_px))
                    y2_px = max(0, min(orig_h, y2_px))

                    if x2_px > x1_px and y2_px > y1_px:
                        boxes_xyxy_pixels.append([x1_px, y1_px, x2_px, y2_px])
                        labels_list.append(cls_id)

        image = self.transform(image_raw)

        # Масштабируем боксы под новый размер
        scale_w = self.img_size / orig_w
        scale_h = self.img_size / orig_h

        boxes_resized = []
        for x1, y1, x2, y2 in boxes_xyxy_pixels:
            boxes_resized.append([
                x1 * scale_w,
                y1 * scale_h,
                x2 * scale_w,
                y2 * scale_h
            ])

        if len(boxes_resized) == 0:
            boxes_tensor = torch.empty((0, 4), dtype=torch.float32)
            labels_tensor = torch.empty((0,), dtype=torch.long)
        else:
            boxes_tensor = torch.tensor(boxes_resized, dtype=torch.float32)
            labels_tensor = torch.tensor(labels_list, dtype=torch.long)

        return {
            'pixel_values': image,
            'targets': {
                'boxes': boxes_tensor,
                'labels': labels_tensor
            }
        }

class MinecraftDatasetFasterRCNN(Dataset):
    """Датасет для Faster R-CNN: возвращает боксы в пикселях."""

    def __init__(self, data_dir: str, split: str = 'train', img_size: int = 640):
        self.data_dir = Path(data_dir)
        self.split = split
        self.img_size = img_size
        self.images_dir = self.data_dir / split / 'images'
        self.labels_dir = self.data_dir / split / 'labels'

        self.files = sorted([
            f.name for f in self.images_dir.glob('*')
            if f.suffix.lower() in ('.jpg', '.jpeg', '.png')
            and (self.labels_dir / (f.stem + '.txt')).exists()
        ])
        # Для теста можно оставить обрезку, для полноценного обучения убрать
        # self.files = self.files[:32]

        logger.info(f"[{split}] Loaded {len(self.files)} images")

        self.transform = T.Compose([
            T.ToTensor(),
            T.Resize((img_size, img_size)),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_file = self.files[idx]
        img_path = self.images_dir / img_file
        label_path = self.labels_dir / (Path(img_file).stem + '.txt')

        image_raw = cv2.imread(str(img_path))
        image_raw = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = image_raw.shape[:2]

        boxes_xyxy_pixels = []
        labels_list = []

        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cls_id = int(parts[0])
                    cx, cy, w, h = map(float, parts[1:5])

                    # Защита от некорректных значений
                    w = max(1e-6, min(1.0, w))
                    h = max(1e-6, min(1.0, h))
                    cx = max(0.0, min(1.0, cx))
                    cy = max(0.0, min(1.0, cy))

                    # Считаем координаты в пикселях на оригинальном изображении
                    x1_px = (cx - w / 2.0) * orig_w
                    y1_px = (cy - h / 2.0) * orig_h
                    x2_px = (cx + w / 2.0) * orig_w
                    y2_px = (cy + h / 2.0) * orig_h

                    # Обрезаем по границам
                    x1_px = max(0, x1_px)
                    y1_px = max(0, y1_px)
                    x2_px = min(orig_w, x2_px)
                    y2_px = min(orig_h, y2_px)

                    if x2_px <= x1_px or y2_px <= y1_px:
                        continue

                    boxes_xyxy_pixels.append([x1_px, y1_px, x2_px, y2_px])
                    labels_list.append(cls_id)

        # Применяем трансформации к изображению
        image = self.transform(image_raw)

        # Масштабируем боксы под новый размер (img_size x img_size)
        new_size = self.img_size
        scale_w = new_size / orig_w
        scale_h = new_size / orig_h

        boxes_resized = []
        for x1, y1, x2, y2 in boxes_xyxy_pixels:
            boxes_resized.append([
                x1 * scale_w,
                y1 * scale_h,
                x2 * scale_w,
                y2 * scale_h
            ])

        # Если объектов нет — возвращаем пустые тензоры правильной размерности
        if len(boxes_resized) == 0:
            boxes_tensor = torch.empty((0, 4), dtype=torch.float32)
            labels_tensor = torch.empty((0,), dtype=torch.long)
        else:
            boxes_tensor = torch.tensor(boxes_resized, dtype=torch.float32)
            labels_tensor = torch.tensor(labels_list, dtype=torch.long)

        return {
            'pixel_values': image,
            'targets': {
                'boxes': boxes_tensor,
                'labels': labels_tensor
            },
            'orig_size': (orig_h, orig_w),
            'image_path': str(img_path)
        }

def get_dataloader(
        config: Dict,
        split: str = 'train',
        batch_size: int = None,
        num_workers: int = None,
        shuffle: bool = None,
        use_mosaic: bool = False
) -> DataLoader:
    """
    Создание DataLoader с расширенными опциями

    Args:
        config: Конфигурация данных
        split: 'train', 'val', 'test'
        batch_size: Размер батча (переопределяет конфиг)
        num_workers: Количество workers
        shuffle: Перемешивать ли данные
        use_mosaic: Использовать Mosaic augmentation

    Returns:
        DataLoader
    """
    # Параметры из конфига
    data_dir = config.get('raw_path', config.get('data_dir', 'data/raw/minecraft_mobs'))
    batch_size = batch_size or config.get('batch_size', 16)
    num_workers = num_workers or config.get('num_workers', 4)
    img_size = config.get('image_size', 640)
    num_classes = config.get('num_classes', 6)
    class_names = config.get('class_names', None)

    if shuffle is None:
        shuffle = (split == 'train')

    # Создание датасета
    dataset = MinecraftMobsDataset(
        data_dir=data_dir,
        split=split,
        img_size=img_size,
        num_classes=num_classes,
        class_names=class_names,
        use_mosaic=use_mosaic and split == 'train',
        mosaic_prob=config.get('augmentation', {}).get('mosaic', 0.5),
        use_mixup=config.get('augmentation', {}).get('mixup', False),
        mixup_prob=config.get('augmentation', {}).get('mixup_prob', 0.1)
    )

    # Создание DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=(split == 'train'),  # Отбрасываем неполный последний батч при тренировке
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=True if num_workers > 0 else False
    )

    logger.info(f"Created {split} DataLoader with {len(dataset)} samples, "
                f"batch_size={batch_size}, workers={num_workers}")

    return dataloader


def collate_fn(batch: List[Dict]) -> Dict:
    """
    Функция объединения батча для поддержки разного количества объектов

    Args:
        batch: Список словарей от Dataset.__getitem__

    Returns:
        Dict с объединенным батчем
    """
    images = torch.stack([item['image'] for item in batch])
    boxes = [item['boxes'] for item in batch]
    labels = [item['labels'] for item in batch]
    image_ids = [item['image_id'] for item in batch]
    orig_sizes = [item['orig_size'] for item in batch]
    image_paths = [item['image_path'] for item in batch]

    return {
        'image': images,
        'boxes': boxes,
        'labels': labels,
        'image_id': image_ids,
        'orig_size': orig_sizes,
        'image_path': image_paths
    }

def collate_fn_faster_rcnn(batch):
    pixel_values = torch.stack([b['pixel_values'] for b in batch])
    orig_size = [b['orig_size'] for b in batch]
    image_path = [b['image_path'] for b in batch]

    targets = [b['targets'] for b in batch]

    return {
        'pixel_values': pixel_values,
        'targets': targets,
        'orig_size': orig_size,
        'image_path': image_path
    }

def collate_fn_rtdetr(batch):
    pixel_values = torch.stack([b['pixel_values'] for b in batch])
    targets = [b['targets'] for b in batch]
    return {
        'pixel_values': pixel_values,
        'targets': targets
    }

def create_balanced_sampler(dataset: MinecraftMobsDataset) -> WeightedRandomSampler:
    """
    Создание сэмплера для балансировки классов

    Args:
        dataset: Экземпляр датасета

    Returns:
        WeightedRandomSampler
    """
    stats = dataset.get_dataset_stats()

    # Вычисляем веса для балансировки
    class_weights = {}
    total_objects = stats['num_objects']

    for class_name, count in stats['class_distribution'].items():
        if count > 0:
            class_weights[class_name] = total_objects / (len(stats['class_distribution']) * count)
        else:
            class_weights[class_name] = 1.0

    # Назначаем вес каждому изображению
    weights = []
    for idx in range(len(dataset)):
        img_file = dataset.image_files[idx]
        label_path = dataset._get_label_path(img_file)

        weight = 1.0
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        if 0 <= class_id < dataset.num_classes:
                            class_name = dataset.class_names[class_id]
                            weight = max(weight, class_weights.get(class_name, 1.0))

        weights.append(weight)

    return WeightedRandomSampler(weights, len(weights), replacement=True)

def load_data_detr(data_dir, split='train', img_size=320, max_images=None):
    """Загружает изображения и аннотации в формате YOLO с фильтрацией"""
    images_dir = Path(data_dir) / split / 'images'
    labels_dir = Path(data_dir) / split / 'labels'

    files = []
    for img_path in images_dir.glob('*'):
        if img_path.suffix.lower() not in ('.jpg', '.jpeg', '.png'):
            continue
        label_path = labels_dir / (img_path.stem + '.txt')
        if label_path.exists():
            files.append(img_path.name)

    # ====== ФИЛЬТРАЦИЯ: только валидные боксы ======
    valid_files = []
    for img_file in files:
        label_path = labels_dir / (Path(img_file).stem + '.txt')
        has_valid_boxes = False
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        try:
                            cls_id = int(parts[0])
                            cx, cy, w, h = map(float, parts[1:5])
                            if 0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0 and w > 0.01 and h > 0.01:
                                has_valid_boxes = True
                                break
                        except:
                            continue
        if has_valid_boxes:
            valid_files.append(img_file)

    files = valid_files
    if max_images is not None:
        files = files[:max_images]

    print(f"[{split}] Загружено {len(files)} изображений (после фильтрации)")

    transform = T.Compose([
        T.ToTensor(),
        T.Resize((img_size, img_size)),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    def get_item(idx):
        img_file = files[idx]
        img_path = images_dir / img_file
        label_path = labels_dir / (Path(img_file).stem + '.txt')

        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = transform(image)

        boxes = []
        labels = []
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    try:
                        cls_id = int(parts[0])
                        cx, cy, w, h = map(float, parts[1:5])
                        if 0 < w <= 1 and 0 < h <= 1 and 0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0:
                            boxes.append([cx, cy, w, h])
                            labels.append(cls_id)
                    except:
                        continue

        if len(boxes) == 0:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.long)
        else:
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
            labels_tensor = torch.tensor(labels, dtype=torch.long)

        return image, boxes_tensor, labels_tensor

    return files, get_item

def make_loader_detr(data_dir, split='train', batch_size=2, img_size=320, max_images=None):
    files, get_item = load_data_detr(data_dir, split, img_size, max_images)

    class SimpleDataset:
        def __len__(self):
            return len(files)
        def __getitem__(self, idx):
            image, boxes, labels = get_item(idx)
            return {'pixel_values': image, 'boxes': boxes, 'labels': labels}

    def collate(batch):
        return {
            'pixel_values': torch.stack([b['pixel_values'] for b in batch]),
            'boxes': [b['boxes'] for b in batch],
            'labels': [b['labels'] for b in batch]
        }

    return DataLoader(SimpleDataset(), batch_size=batch_size, shuffle=True, collate_fn=collate, num_workers=0)
