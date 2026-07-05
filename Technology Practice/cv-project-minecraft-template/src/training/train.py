"""
Модуль обучения моделей детекции объектов.
Поддерживает обучение YOLO, Faster R-CNN, DETR, RT-DETR, YOLO-World.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    ReduceLROnPlateau,
    StepLR,
    OneCycleLR
)
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging
import time
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Ранняя остановка обучения"""

    def __init__(
            self,
            patience: int = 10,
            min_delta: float = 0.001,
            mode: str = 'max'
    ):
        """
        Args:
            patience: Количество эпох без улучшения
            min_delta: Минимальное изменение для считания улучшением
            mode: 'max' для метрик (mAP), 'min' для loss
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

        if mode == 'max':
            self.best_score = -float('inf')
            self.compare = lambda x, y: x > y + min_delta
        else:
            self.best_score = float('inf')
            self.compare = lambda x, y: x < y - min_delta

    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
        elif self.compare(score, self.best_score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop

    def reset(self):
        self.counter = 0
        self.early_stop = False


class ModelCheckpoint:
    """Сохранение лучших чекпоинтов"""

    def __init__(self, save_dir: Path, model_name: str):
        self.save_dir = Path(save_dir)
        self.model_name = model_name
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.best_metrics = {}

    def save(
            self,
            model: Any,
            optimizer: optim.Optimizer,
            epoch: int,
            metrics: Dict[str, float],
            is_best: bool = False,
            filename: str = None
    ) -> str:
        """
        Сохранение чекпоинта

        Args:
            model: Модель
            optimizer: Оптимизатор
            epoch: Номер эпохи
            metrics: Метрики
            is_best: Лучшая ли модель
            filename: Имя файла

        Returns:
            Путь к сохраненному файлу
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict() if hasattr(model, 'state_dict') else None,
            'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }

        if filename:
            path = self.save_dir / filename
        else:
            path = self.save_dir / f"{self.model_name}_epoch_{epoch}.pth"

        torch.save(checkpoint, path)

        if is_best:
            best_path = self.save_dir / f"{self.model_name}_best.pth"
            torch.save(checkpoint, best_path)
            logger.info(f"Best model saved to {best_path}")

        return str(path)


def create_optimizer(
        model: nn.Module,
        config: Dict
) -> optim.Optimizer:
    """
    Создание оптимизатора с разными learning rates для разных частей модели

    Args:
        model: Модель
        config: Конфигурация

    Returns:
        Оптимизатор
    """
    optimizer_name = config.get('optimizer', 'adamw').lower()
    lr = config.get('lr', 0.001)
    weight_decay = config.get('weight_decay', 0.0005)

    # Разделяем параметры на группы (backbone vs head)
    if hasattr(model, 'backbone') and hasattr(model, 'head'):
        backbone_params = []
        head_params = []

        for name, param in model.named_parameters():
            if 'backbone' in name:
                backbone_params.append(param)
            else:
                head_params.append(param)

        param_groups = [
            {'params': backbone_params, 'lr': lr * 0.1},  # Backbone учим медленнее
            {'params': head_params, 'lr': lr}
        ]
    else:
        param_groups = model.parameters()

    # Создаем оптимизатор
    if optimizer_name == 'adam':
        return optim.Adam(param_groups, lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'adamw':
        return optim.AdamW(param_groups, lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'sgd':
        momentum = config.get('momentum', 0.9)
        return optim.SGD(param_groups, lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


def create_scheduler(
        optimizer: optim.Optimizer,
        config: Dict,
        epochs: int
) -> optim.lr_scheduler._LRScheduler:
    """
    Создание планировщика learning rate

    Args:
        optimizer: Оптимизатор
        config: Конфигурация
        epochs: Количество эпох

    Returns:
        Планировщик
    """
    scheduler_name = config.get('lr_scheduler', 'cosine').lower()

    if scheduler_name == 'cosine':
        return CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    elif scheduler_name == 'step':
        step_size = config.get('lr_step_size', 30)
        gamma = config.get('lr_gamma', 0.1)
        return StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_name == 'plateau':
        return ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=5,
            verbose=True
        )
    elif scheduler_name == 'one_cycle':
        return OneCycleLR(
            optimizer,
            max_lr=config.get('lr', 0.001),
            epochs=epochs,
            steps_per_epoch=100  # примерно
        )
    else:
        return None


def train_one_epoch(
        model: Any,
        train_loader: torch.utils.data.DataLoader,
        optimizer: optim.Optimizer,
        epoch: int,
        device: torch.device,
        scaler: Optional[GradScaler] = None,
        use_amp: bool = False,
        grad_clip: float = None
) -> Dict[str, float]:
    """
    Обучение одной эпохи

    Args:
        model: Модель
        train_loader: DataLoader с тренировочными данными
        optimizer: Оптимизатор
        epoch: Номер эпохи
        device: Устройство
        scaler: GradScaler для mixed precision
        use_amp: Использовать ли mixed precision
        grad_clip: Максимальная норма градиентов

    Returns:
        Словарь со средними значениями лоссов
    """
    model.train()

    epoch_losses = {
        'total_loss': [],
        'cls_loss': [],
        'box_loss': [],
        'obj_loss': []
    }

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]", leave=False)

    for batch_idx, batch in enumerate(progress_bar):
        # Перемещаем данные на устройство
        images = batch['image'].to(device)
        targets = {
            'boxes': [b.to(device) for b in batch['boxes']],
            'labels': [l.to(device) for l in batch['labels']]
        }

        optimizer.zero_grad()

        # Mixed precision training
        if use_amp and scaler:
            with autocast():
                loss_dict = model(images, targets)

                if isinstance(loss_dict, dict):
                    total_loss = sum(loss for loss in loss_dict.values())
                else:
                    total_loss = loss_dict

            scaler.scale(total_loss).backward()

            if grad_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            scaler.step(optimizer)
            scaler.update()
        else:
            # Обычное обучение
            loss_dict = model(images, targets)

            if isinstance(loss_dict, dict):
                total_loss = sum(loss for loss in loss_dict.values())
            else:
                total_loss = loss_dict

            total_loss.backward()

            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()

        # Логирование лоссов
        epoch_losses['total_loss'].append(total_loss.item())

        if isinstance(loss_dict, dict):
            for key, value in loss_dict.items():
                if key in epoch_losses:
                    epoch_losses[key].append(value.item())

        # Обновляем прогресс-бар
        progress_bar.set_postfix({
            'loss': f"{total_loss.item():.4f}",
            'lr': f"{optimizer.param_groups[0]['lr']:.6f}"
        })

    # Усредняем лоссы
    avg_losses = {
        key: np.mean(values) if values else 0.0
        for key, values in epoch_losses.items()
    }

    return avg_losses


@torch.no_grad()
def validate(
        model: Any,
        val_loader: torch.utils.data.DataLoader,
        device: torch.device,
        class_names: List[str] = None
) -> Dict[str, float]:
    """
    Валидация модели

    Args:
        model: Модель
        val_loader: DataLoader с валидационными данными
        device: Устройство
        class_names: Имена классов

    Returns:
        Словарь с метриками валидации
    """
    model.eval()

    val_losses = {
        'total_loss': [],
        'cls_loss': [],
        'box_loss': []
    }

    all_predictions = []
    all_targets = []

    progress_bar = tqdm(val_loader, desc="Validation", leave=False)

    for batch in progress_bar:
        images = batch['image'].to(device)
        targets = {
            'boxes': [b.to(device) for b in batch['boxes']],
            'labels': [l.to(device) for l in batch['labels']]
        }

        # Вычисляем loss
        loss_dict = model(images, targets)

        if isinstance(loss_dict, dict):
            total_loss = sum(loss for loss in loss_dict.values())
        else:
            total_loss = loss_dict

        val_losses['total_loss'].append(total_loss.item())

        if isinstance(loss_dict, dict):
            for key, value in loss_dict.items():
                if key in val_losses:
                    val_losses[key].append(value.item())

        # Получаем предсказания
        predictions = model(images)

        # Конвертируем для метрик
        for i in range(len(images)):
            pred = {
                'boxes': predictions[i]['boxes'].cpu().numpy() if len(predictions[i]['boxes']) > 0 else [],
                'scores': predictions[i]['scores'].cpu().numpy() if len(predictions[i]['scores']) > 0 else [],
                'labels': predictions[i]['labels'].cpu().numpy() if len(predictions[i]['labels']) > 0 else []
            }
            target = {
                'boxes': targets['boxes'][i].cpu().numpy() if len(targets['boxes'][i]) > 0 else [],
                'labels': targets['labels'][i].cpu().numpy() if len(targets['labels'][i]) > 0 else []
            }

            all_predictions.append(pred)
            all_targets.append(target)

    # Вычисляем метрики (упрощенно, основное в metrics.py)
    from src.evaluation.metrics import calculate_map, calculate_precision_recall

    map_metrics = calculate_map(
        predictions=all_predictions,
        targets=all_targets,
        num_classes=len(class_names) if class_names else 6
    )

    pr_metrics = calculate_precision_recall(all_predictions, all_targets)

    # Усредняем лоссы
    avg_losses = {
        key: np.mean(values) if values else 0.0
        for key, values in val_losses.items()
    }

    # Объединяем все метрики
    metrics = {
        **avg_losses,
        'mAP_50': map_metrics.get('mAP_50', 0.0),
        'mAP_50_95': map_metrics.get('mAP_50_95', 0.0),
        'precision': pr_metrics.get('precision', 0.0),
        'recall': pr_metrics.get('recall', 0.0),
        'f1_score': pr_metrics.get('f1_score', 0.0)
    }

    return metrics


def train_model(
        model: Any,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        config: Dict,
        model_name: str,
        save_dir: Optional[Path] = None
) -> Dict[str, List]:
    """
    Полный цикл обучения модели

    Args:
        model: Модель детекции
        train_loader: DataLoader с тренировочными данными
        val_loader: DataLoader с валидационными данными
        config: Конфигурация обучения
        model_name: Название модели
        save_dir: Директория для сохранения

    Returns:
        История обучения
    """
    # Параметры обучения
    epochs = config.get('epochs', 100)
    device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    use_amp = config.get('mixed_precision', True) and device.type == 'cuda'
    grad_clip = config.get('grad_clip', None)
    seed = config.get('seed', 42)

    # Устанавливаем seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)

    # Перемещаем модель на устройство
    model = model.to(device)

    # Создаем оптимизатор и планировщик
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config, epochs)

    # Mixed precision
    scaler = GradScaler() if use_amp else None

    # Callbacks
    early_stopping = EarlyStopping(
        patience=config.get('patience', 10),
        min_delta=config.get('early_stopping', {}).get('min_delta', 0.001),
        mode='max'  # максимизируем mAP
    )

    checkpoint = ModelCheckpoint(
        save_dir=save_dir or Path('results/checkpoints'),
        model_name=model_name
    )

    # TensorBoard
    writer = SummaryWriter(
        log_dir=f"results/logs/{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    # История обучения
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_cls_loss': [],
        'train_box_loss': [],
        'val_cls_loss': [],
        'val_box_loss': [],
        'mAP_50': [],
        'mAP_50_95': [],
        'precision': [],
        'recall': [],
        'f1_score': [],
        'learning_rate': []
    }

    logger.info(f"Starting training {model_name} on {device}")
    logger.info(f"Epochs: {epochs}, Batch size: {train_loader.batch_size}")
    logger.info(f"Train samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}")
    logger.info(f"Mixed precision: {use_amp}, Gradient clipping: {grad_clip}")

    start_time = time.time()

    # Основной цикл обучения
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        # Обучение одной эпохи
        train_losses = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            epoch=epoch,
            device=device,
            scaler=scaler,
            use_amp=use_amp,
            grad_clip=grad_clip
        )

        # Валидация
        val_metrics = validate(
            model=model,
            val_loader=val_loader,
            device=device,
            class_names=config.get('class_names', None)
        )

        # Обновляем learning rate
        current_lr = optimizer.param_groups[0]['lr']

        if scheduler:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_metrics['mAP_50'])
            else:
                scheduler.step()

        # Сохраняем в историю
        history['train_loss'].append(train_losses['total_loss'])
        history['val_loss'].append(val_metrics['total_loss'])
        history['train_cls_loss'].append(train_losses.get('cls_loss', 0))
        history['train_box_loss'].append(train_losses.get('box_loss', 0))
        history['val_cls_loss'].append(val_metrics.get('cls_loss', 0))
        history['val_box_loss'].append(val_metrics.get('box_loss', 0))
        history['mAP_50'].append(val_metrics['mAP_50'])
        history['mAP_50_95'].append(val_metrics['mAP_50_95'])
        history['precision'].append(val_metrics['precision'])
        history['recall'].append(val_metrics['recall'])
        history['f1_score'].append(val_metrics['f1_score'])
        history['learning_rate'].append(current_lr)

        # Логирование в TensorBoard
        writer.add_scalar('Loss/train', train_losses['total_loss'], epoch)
        writer.add_scalar('Loss/val', val_metrics['total_loss'], epoch)
        writer.add_scalar('Metrics/mAP_50', val_metrics['mAP_50'], epoch)
        writer.add_scalar('Metrics/mAP_50_95', val_metrics['mAP_50_95'], epoch)
        writer.add_scalar('Metrics/precision', val_metrics['precision'], epoch)
        writer.add_scalar('Metrics/recall', val_metrics['recall'], epoch)
        writer.add_scalar('LR', current_lr, epoch)

        if 'cls_loss' in train_losses:
            writer.add_scalar('Loss/train_cls', train_losses['cls_loss'], epoch)
        if 'box_loss' in train_losses:
            writer.add_scalar('Loss/train_box', train_losses['box_loss'], epoch)

        # Время эпохи
        epoch_time = time.time() - epoch_start

        # Вывод метрик
        logger.info(
            f"Epoch {epoch:3d}/{epochs} | "
            f"Train Loss: {train_losses['total_loss']:.4f} | "
            f"Val Loss: {val_metrics['total_loss']:.4f} | "
            f"mAP@0.5: {val_metrics['mAP_50']:.4f} | "
            f"mAP@0.5:0.95: {val_metrics['mAP_50_95']:.4f} | "
            f"LR: {current_lr:.6f} | "
            f"Time: {epoch_time:.1f}s"
        )

        # Сохраняем лучшую модель
        is_best = val_metrics['mAP_50'] >= max(history['mAP_50']) if history['mAP_50'] else True
        checkpoint.save(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            metrics=val_metrics,
            is_best=is_best
        )

        # Early stopping
        if early_stopping(val_metrics['mAP_50']):
            logger.info(f"Early stopping triggered at epoch {epoch}")
            break

    # Завершение
    total_time = time.time() - start_time
    logger.info(f"Training completed in {total_time / 60:.1f} minutes")
    logger.info(f"Best mAP@0.5: {max(history['mAP_50']):.4f}")

    writer.close()

    # Сохраняем историю обучения
    history_path = save_dir / f"{model_name}_history.json" if save_dir else Path(
        f"results/logs/{model_name}_history.json")
    history_path.parent.mkdir(parents=True, exist_ok=True)

    # Конвертируем numpy типы для JSON
    history_json = {
        key: [float(v) for v in values]
        for key, values in history.items()
    }

    with open(history_path, 'w') as f:
        json.dump(history_json, f, indent=2)

    return history


def train_yolo(
        model: Any,
        config: Dict,
        model_name: str,
        data_yaml_path: str = None
) -> Dict:
    """
    Специальная функция для обучения YOLO моделей

    YOLO имеет встроенный метод train с особым форматом данных

    Args:
        model: YOLO модель
        config: Конфигурация
        model_name: Название модели
        data_yaml_path: Путь к YAML с данными

    Returns:
        История обучения
    """
    logger.info(f"Training YOLO model: {model_name}")

    # Параметры обучения
    train_args = {
        'data': data_yaml_path or 'data/data.yaml',
        'epochs': config.get('epochs', 100),
        'imgsz': config.get('image_size', 640),
        'batch': config.get('batch_size', 16),
        'lr0': config.get('lr', 0.001),
        'lrf': config.get('lrf', 0.01),
        'device': config.get('device', 'cuda'),
        'workers': config.get('num_workers', 4),
        'project': 'results/logs',
        'name': f'yolo_{model_name}',
        'exist_ok': True,
        'pretrained': config.get('pretrained', True),
        'optimizer': config.get('optimizer', 'AdamW'),
        'seed': config.get('seed', 42),
        'patience': config.get('patience', 10),
        'save': True,
        'plots': True,
        'verbose': True
    }

    # Запуск обучения
    results = model.train(**train_args)

    # Извлечение истории
    history = {
        'train_loss': results.results_dict.get('train/box_loss', []),
        'val_loss': results.results_dict.get('val/box_loss', []),
        'mAP_50': results.results_dict.get('metrics/mAP50(B)', 0.0),
        'mAP_50_95': results.results_dict.get('metrics/mAP50-95(B)', 0.0),
        'precision': results.results_dict.get('metrics/precision(B)', 0.0),
        'recall': results.results_dict.get('metrics/recall(B)', 0.0)
    }

    logger.info(f"YOLO training completed. mAP@0.5: {history['mAP_50']:.4f}")

    return history
