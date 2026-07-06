"""
trainer.py - Модуль для запуска обучения моделей
"""
import os

import yaml
from pathlib import Path
from typing import List, Optional, Dict, Any
from ..dataset.dataset import MinecraftMobsDataset, MinecraftDatasetFasterRCNN, collate_fn_faster_rcnn, make_loader_detr
from ..models.factory import create_detector
from ..utils.utils import load_config
from ..training.train import train_faster_rcnn
from torch.utils.data import DataLoader
import torch
from datetime import datetime
from ..training.train import train_deformable_detr

def prepare_dataset(config: dict, model_name: str) -> str:
    """
    Подготавливает датасет и создаёт data.yaml для YOLO-моделей.
    """
    class_names = config['data']['class_names']

    box_format = 'yolo'
    if model_name in ['detr', 'deformable_detr']:
        box_format = 'cxcywh'
    elif model_name == 'faster_rcnn':
        box_format = 'pixel_xyxy_orig'

    dataset = MinecraftMobsDataset(
        data_dir=config['data']['raw_path'],
        split='train',
        img_size=config['data']['image_size'],
        box_format=box_format,
        class_names=class_names,
        num_classes=len(class_names)
    )

    data_yaml_path = 'data.yaml'
    dataset.create_data_yaml(data_yaml_path)

    return data_yaml_path


def train_model(
    model_name: str,
    config: Dict[str, Any],
    data_yaml_path: Optional[str] = None,
    override_params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Обучает одну модель.
    """
    class_names = config['data']['class_names']

    if data_yaml_path is None:
        data_yaml_path = prepare_dataset(config, model_name)

    model_config = config['models'].get(model_name, {})
    if override_params:
        model_config.update(override_params)

    detector = create_detector(model_name, model_config, class_names)

    # ====== YOLO / YOLO-World ======
    if model_name in ['yolo', 'yolo_world']:
        results = detector.train(
            train_loader=None,
            val_loader=None,
            data_yaml_path=data_yaml_path
        )

    elif model_name in ['rtdetr', 'rt_detr']:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        project_root = os.getcwd()
        save_dir = os.path.join(project_root, 'results', f'rtdetr_{timestamp}')
        results = detector.train(
            data_yaml_path=data_yaml_path,
            epochs=model_config.get('epochs', 50),
            imgsz=model_config.get('image_size', config['data']['image_size']),
            batch=model_config.get('batch_size', config['data']['batch_size']),
            lr0=model_config.get('lr', 0.001),
            device=model_config.get('device', config['training'].get('device', 'cpu')),
            project=save_dir,
            name='',
            exist_ok=True,
            verbose=True
        )

    elif model_name == 'faster_rcnn':
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join('results', f'faster_rcnn_{timestamp}')
        os.makedirs(save_dir, exist_ok=True)

        # Создаём датасеты
        train_dataset = MinecraftDatasetFasterRCNN(
            data_dir=config['data']['raw_path'],
            split='train',
            img_size=model_config.get('image_size', config['data']['image_size'])
        )

        val_dataset = MinecraftDatasetFasterRCNN(
            data_dir=config['data']['raw_path'],
            split='val',
            img_size=model_config.get('image_size', config['data']['image_size'])
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=model_config.get('batch_size', config['data']['batch_size']),
            shuffle=True,
            collate_fn=collate_fn_faster_rcnn,
            num_workers=0
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=model_config.get('batch_size', config['data']['batch_size']),
            shuffle=False,
            collate_fn=collate_fn_faster_rcnn,
            num_workers=0
        )
        detector.build_model()
        history = train_faster_rcnn(
            model=detector.model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=model_config.get('epochs', 50),
            lr=model_config.get('lr', 1e-4),
            weight_decay=model_config.get('weight_decay', 1e-4),
            device=model_config.get('device', 'cpu'),
            save_dir=save_dir
        )

        results = {
            'model_path': save_dir,
            'history': history
        }

    elif model_name in ['detr', 'deformable_detr']:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join('results', f'detr_{timestamp}')
        os.makedirs(save_dir, exist_ok=True)

        # Создаём DataLoader'ы через DETR-загрузчик
        train_loader = make_loader_detr(
            data_dir=config['data']['raw_path'],
            split='train',
            batch_size=model_config.get('batch_size', config['data']['batch_size']),
            img_size=model_config.get('image_size', config['data']['image_size']),
            max_images=model_config.get('max_images', None)  # для теста
        )

        val_loader = make_loader_detr(
            data_dir=config['data']['raw_path'],
            split='val',
            batch_size=model_config.get('batch_size', config['data']['batch_size']),
            img_size=model_config.get('image_size', config['data']['image_size']),
            max_images=model_config.get('max_images', None)
        )

        detector.build_model()
        history = train_deformable_detr(
            model=detector.model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=model_config.get('epochs', 50),
            lr=model_config.get('lr', 1e-4),
            device=model_config.get('device', 'cpu'),
            save_dir=save_dir
        )

        results = {
            'model_path': save_dir,
            'history': history
        }

    else:
        box_format = 'cxcywh' if 'detr' in model_name else 'pixel_xyxy_orig'

        train_dataset = MinecraftMobsDataset(
            data_dir=config['data']['raw_path'],
            split='train',
            img_size=model_config.get('data',config['data']['image_size']),
            box_format=box_format,
            class_names=class_names,
            num_classes=len(class_names)
        )

        val_dataset = MinecraftMobsDataset(
            data_dir=config['data']['raw_path'],
            split='val',
            img_size=config['data']['image_size'],
            box_format=box_format,
            class_names=class_names,
            num_classes=len(class_names)
        )

        def collate_fn(batch):
            pixel_values = torch.stack([b['pixel_values'] for b in batch])
            labels = {
                'boxes': [b['labels']['boxes'] for b in batch],
                'labels': [b['labels']['labels'] for b in batch]
            }
            return {'pixel_values': pixel_values, 'labels': labels}

        train_loader = DataLoader(
            train_dataset,
            batch_size=config['data']['batch_size'],
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=config['data']['batch_size'],
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0
        )

        results = detector.train(train_loader, val_loader)

    print(f"Обучение {model_name} завершено!")
    return results


def train_models(
    model_names: List[str],
    config_path: str = 'configs/default.yaml',
    override_params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Обучает несколько моделей из списка.
    """
    config = load_config(config_path)

    data_yaml_path = None
    for model_name in model_names:
        if model_name in ['yolo', 'yolo_world', 'rtdetr', 'rt_detr']:
            if data_yaml_path is None:
                data_yaml_path = prepare_dataset(config, model_name)

    results = {}
    for model_name in model_names:
        print(f"\nЗапуск обучения {model_name}...")
        results[model_name] = train_model(
            model_name,
            config,
            data_yaml_path=data_yaml_path if model_name in ['yolo', 'yolo_world', 'rtdetr', 'rt_detr'] else None,
            override_params=override_params
        )

    return results
