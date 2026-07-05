"""
Модуль для загрузки и предобработки датасета Minecraft Mobs.
Поддерживает форматы: YOLO, COCO, Pascal VOC.
"""

import os
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Callable
import logging

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
            num_classes: int = 6,
            class_names: List[str] = None,
            use_mosaic: bool = False,
            mosaic_prob: float = 0.5,
            use_mixup: bool = False,
            mixup_prob: float = 0.1,
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
            use_mosaic: Использовать Mosaic augmentation
            mosaic_prob: Вероятность применения mosaic
            use_mixup: Использовать MixUp augmentation
            mixup_prob: Вероятность применения mixup
            debug: Режим отладки (меньше логов, больше проверок)
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.img_size = img_size
        self.num_classes = num_classes
        self.class_names = class_names or [f"class_{i}" for i in range(num_classes)]
        self.use_mosaic = use_mosaic and split == 'train'
        self.mosaic_prob = mosaic_prob
        self.use_mixup = use_mixup and split == 'train'
        self.mixup_prob = mixup_prob
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

    def _get_default_transforms(self) -> Callable:
        """Создание дефолтных трансформаций в зависимости от сплита"""
        if self.split == 'train':
            return A.Compose([
                # Ресайз
                A.Resize(self.img_size, self.img_size),

                # Аугментации для разнообразия
                A.HueSaturationValue(
                    hue_shift_limit=15,
                    sat_shift_limit=30,
                    val_shift_limit=20,
                    p=0.5
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=0.5
                ),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),

                # Геометрические аугментации
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.1),
                A.Rotate(limit=10, border_mode=cv2.BORDER_CONSTANT, p=0.3),
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.1,
                    rotate_limit=0,
                    border_mode=cv2.BORDER_CONSTANT,
                    p=0.3
                ),

                # Blur для устойчивости
                A.OneOf([
                    A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                    A.MedianBlur(blur_limit=5, p=1.0),
                    A.MotionBlur(blur_limit=5, p=1.0),
                ], p=0.2),

                # Нормализация
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2(),
            ], bbox_params=A.BboxParams(
                format='yolo',
                min_visibility=0.3,
                label_fields=['class_labels']
            ))
        else:
            # Валидация/тест - только базовые трансформации
            return A.Compose([
                A.Resize(self.img_size, self.img_size),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2(),
            ], bbox_params=A.BboxParams(
                format='yolo',
                min_visibility=0.3,
                label_fields=['class_labels']
            ))

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Dict:
        """
        Возвращает элемент датасета

        Returns:
            Dict с ключами:
                - 'image': torch.Tensor (C, H, W)
                - 'boxes': torch.Tensor (N, 4) в формате [x1, y1, x2, y2] (пиксели)
                - 'labels': torch.Tensor (N,)
                - 'image_id': int
                - 'orig_size': Tuple (H, W)
                - 'image_path': str
        """
        # Mosaic augmentation (только для тренировки)
        if self.use_mosaic and random.random() < self.mosaic_prob:
            return self._get_mosaic_item(idx)

        # Загрузка изображения и аннотаций
        img_file = self.image_files[idx]
        img_path = self.images_dir / img_file
        label_path = self._get_label_path(img_file)

        # Загрузка изображения
        image = cv2.imread(str(img_path))
        if image is None:
            raise RuntimeError(f"Failed to load image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_size = image.shape[:2]

        # Загрузка аннотаций
        boxes_yolo, labels = self._load_yolo_annotations(label_path)

        # Применяем аугментации
        if self.transform and len(boxes_yolo) > 0:
            transformed = self.transform(
                image=image,
                bboxes=boxes_yolo,
                class_labels=labels
            )
            image = transformed['image']
            boxes_yolo = transformed['bboxes']
            labels = transformed['class_labels']
        elif self.transform:
            # Если нет боксов, применяем трансформации только к изображению
            transformed = self.transform(
                image=image,
                bboxes=[],
                class_labels=[]
            )
            image = transformed['image']
            boxes_yolo = []
            labels = []

        # Конвертируем YOLO формат в [x1, y1, x2, y2]
        boxes_pixel = self._yolo_to_pixel(boxes_yolo, self.img_size, self.img_size)

        # MixUp augmentation
        if self.use_mixup and random.random() < self.mixup_prob and len(boxes_pixel) > 0:
            mix_idx = random.randint(0, len(self) - 1)
            mix_item = self[mix_idx]

            # Смешиваем изображения и аннотации
            lambda_val = np.random.beta(0.5, 0.5)
            image = lambda_val * image + (1 - lambda_val) * mix_item['image']

            # Объединяем аннотации
            boxes_pixel = torch.cat([boxes_pixel, mix_item['boxes']], dim=0)
            labels = torch.cat([
                torch.tensor(labels),
                mix_item['labels']
            ], dim=0)

        # Создаем тензоры
        boxes_tensor = torch.tensor(boxes_pixel, dtype=torch.float32) if len(boxes_pixel) > 0 else torch.zeros((0, 4),
                                                                                                               dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.long) if len(labels) > 0 else torch.zeros(0, dtype=torch.long)

        sample = {
            'image': image,
            'boxes': boxes_tensor,
            'labels': labels_tensor,
            'image_id': idx,
            'orig_size': orig_size,
            'image_path': str(img_path)
        }

        # Дополнительные трансформации целевых переменных
        if self.target_transform:
            sample = self.target_transform(sample)

        return sample

    def _load_yolo_annotations(self, label_path: Path) -> Tuple[List, List]:
        """
        Загрузка аннотаций в формате YOLO

        YOLO формат: class_id center_x center_y width height (нормализованные)

        Args:
            label_path: Путь к файлу аннотации

        Returns:
            Кортеж (boxes, labels)
        """
        boxes = []
        labels = []

        if label_path.exists():
            try:
                with open(label_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue

                        parts = line.split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            bbox = [float(x) for x in parts[1:5]]

                            # Проверка корректности
                            if all(0 <= x <= 1 for x in bbox) and 0 <= class_id < self.num_classes:
                                boxes.append(bbox)
                                labels.append(class_id)
                            elif self.debug:
                                logger.warning(f"Invalid annotation in {label_path}: {line}")

            except Exception as e:
                logger.error(f"Error loading {label_path}: {e}")

        return boxes, labels

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

    def _get_mosaic_item(self, idx: int) -> Dict:
        """
        Mosaic augmentation - объединение 4 изображений в одно

        Returns:
            Dict с объединенным изображением и аннотациями
        """
        # Выбираем 4 случайных изображения
        indices = [idx] + [random.randint(0, len(self) - 1) for _ in range(3)]

        # Центральная точка соединения
        cx = int(random.uniform(self.img_size // 4, 3 * self.img_size // 4))
        cy = int(random.uniform(self.img_size // 4, 3 * self.img_size // 4))

        mosaic_image = np.full((self.img_size * 2, self.img_size * 2, 3), 114, dtype=np.uint8)
        mosaic_boxes = []
        mosaic_labels = []

        # Позиции для 4 изображений
        positions = [
            (0, 0, cx, cy),  # top-left
            (cx, 0, self.img_size, cy),  # top-right
            (0, cy, cx, self.img_size),  # bottom-left
            (cx, cy, self.img_size, self.img_size)  # bottom-right
        ]

        for i, (mosaic_idx, (x1, y1, x2, y2)) in enumerate(zip(indices, positions)):
            # Загружаем изображение
            img_file = self.image_files[mosaic_idx]
            img_path = self.images_dir / img_file
            label_path = self._get_label_path(img_file)

            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (x2 - x1, y2 - y1))

            # Размещаем в мозаике
            mosaic_image[y1:y2, x1:x2] = img

            # Загружаем аннотации и масштабируем
            boxes_yolo, labels = self._load_yolo_annotations(label_path)

            for box, label in zip(boxes_yolo, labels):
                # Конвертируем YOLO в пиксели для оригинального изображения
                orig_w = x2 - x1
                orig_h = y2 - y1

                cx_box = box[0] * orig_w + x1
                cy_box = box[1] * orig_h + y1
                w = box[2] * orig_w
                h = box[3] * orig_h

                # Конвертируем обратно в YOLO для всего мозаичного изображения
                mosaic_cx = cx_box / (self.img_size * 2)
                mosaic_cy = cy_box / (self.img_size * 2)
                mosaic_w = w / (self.img_size * 2)
                mosaic_h = h / (self.img_size * 2)

                mosaic_boxes.append([mosaic_cx, mosaic_cy, mosaic_w, mosaic_h])
                mosaic_labels.append(label)

        # Ресайзим до нужного размера
        mosaic_image = cv2.resize(mosaic_image, (self.img_size, self.img_size))

        # Применяем оставшиеся трансформации
        if self.transform:
            transformed = self.transform(
                image=mosaic_image,
                bboxes=mosaic_boxes if mosaic_boxes else [],
                class_labels=mosaic_labels if mosaic_labels else []
            )
            mosaic_image = transformed['image']
            mosaic_boxes = transformed['bboxes'] if 'bboxes' in transformed else []
            mosaic_labels = transformed['class_labels'] if 'class_labels' in transformed else []

        # Конвертируем в пиксели
        boxes_pixel = self._yolo_to_pixel(mosaic_boxes, self.img_size, self.img_size)

        return {
            'image': mosaic_image,
            'boxes': torch.tensor(boxes_pixel, dtype=torch.float32),
            'labels': torch.tensor(mosaic_labels, dtype=torch.long),
            'image_id': idx,
            'orig_size': (self.img_size * 2, self.img_size * 2),
            'image_path': 'mosaic'
        }

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
            yaml.dump(data_config, f, default_flow_flow=False, allow_unicode=True)

        logger.info(f"Data YAML created at {output_path}")
        return str(output_path)


class COCOFormatDataset(MinecraftMobsDataset):
    """Датасет с поддержкой COCO формата аннотаций"""

    def _load_coco_annotations(self, annotation_file: str) -> Dict:
        """Загрузка аннотаций в формате COCO"""
        with open(annotation_file, 'r') as f:
            coco_data = json.load(f)

        # Создаем индекс по image_id
        annotations_by_image = defaultdict(list)
        for ann in coco_data['annotations']:
            annotations_by_image[ann['image_id']].append(ann)

        return {
            'images': {img['id']: img for img in coco_data['images']},
            'annotations': annotations_by_image,
            'categories': {cat['id']: cat for cat in coco_data['categories']}
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
