"""
Модуль обучения моделей детекции объектов.
Поддерживает обучение YOLO, Faster R-CNN, DETR, RT-DETR, YOLO-World.
"""
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    ReduceLROnPlateau,
    StepLR,
    OneCycleLR
)
from torchmetrics.detection import MeanAveragePrecision
import os
import yaml
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
from ..utils.visualization import plot_training_history

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
    """
    optimizer_name = config.get('optimizer', 'adamw').lower()
    lr = config.get('lr', 0.001)
    weight_decay = config.get('weight_decay', 0.0005)

    # Получаем параметры (поддерживаем врапперы)
    if hasattr(model, 'model') and hasattr(model.model, 'parameters'):
        actual_model = model.model
    elif hasattr(model, 'parameters'):
        actual_model = model
    else:
        raise AttributeError(f"Model {type(model)} has no parameters()")

    # Разделяем параметры на группы (backbone vs head)
    if hasattr(actual_model, 'backbone') and hasattr(actual_model, 'head'):
        backbone_params = []
        head_params = []

        for name, param in actual_model.named_parameters():
            if 'backbone' in name:
                backbone_params.append(param)
            else:
                head_params.append(param)

        param_groups = [
            {'params': backbone_params, 'lr': lr * 0.1},
            {'params': head_params, 'lr': lr}
        ]
    else:
        param_groups = actual_model.parameters()

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
    if hasattr(model, 'model'):
        model.model.train()
    else:
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
        targets = []
        for boxes, labels in zip(batch['boxes'], batch['labels']):
            targets.append({
                'boxes': boxes.to(device),
                'labels': labels.to(device)
            })

        optimizer.zero_grad()

        # Mixed precision training
        if use_amp and scaler:
            with autocast():
                if hasattr(model, 'model'):
                    loss_dict = model.model(images, targets)
                else:
                    loss_dict = model(images, targets)

                if isinstance(loss_dict, dict):
                    total_loss = sum(loss for loss in loss_dict.values())
                else:
                    total_loss = loss_dict

            scaler.scale(total_loss).backward()

            if grad_clip:
                scaler.unscale_(optimizer)
                params = model.model.parameters() if hasattr(model, 'model') else model.parameters()
                torch.nn.utils.clip_grad_norm_(params, grad_clip)

            scaler.step(optimizer)
            scaler.update()
        else:
            # Обычное обучение
            if hasattr(model, 'model'):
                # DETR из Hugging Face принимает pixel_values и labels
                if isinstance(model.model, torch.nn.Module):
                    try:
                        loss_dict = model.model(images, targets)
                    except:
                        loss_dict = model.model(images, targets)
                else:
                    loss_dict = model.model(images, targets)
            else:
                loss_dict = model(images, targets)

            if hasattr(model, 'model'):
                loss_dict = model.model(images, targets)
            else:
                loss_dict = model(images, targets)

            total_loss.backward()

            if grad_clip:
                params = model.model.parameters() if hasattr(model, 'model') else model.parameters()
                torch.nn.utils.clip_grad_norm_(params, grad_clip)

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
    """Валидация модели"""

    if hasattr(model, 'model'):
        model.model.eval()
    else:
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

        # Подготавливаем targets как список словарей
        batch_targets = []
        for boxes, labels in zip(batch['boxes'], batch['labels']):
            batch_targets.append({
                'boxes': boxes.to(device),
                'labels': labels.to(device)
            })

        # Вычисляем loss
        if hasattr(model, 'model'):
            loss_dict = model.model(images, batch_targets)
        else:
            loss_dict = model(images, batch_targets)

        if isinstance(loss_dict, dict):
            total_loss = sum(loss for loss in loss_dict.values())
        else:
            total_loss = loss_dict

        val_losses['total_loss'].append(total_loss.item())

        if isinstance(loss_dict, dict):
            for key, value in loss_dict.items():
                if key in val_losses:
                    val_losses[key].append(value.item())

        # Получаем предсказания (в eval режиме)
        if hasattr(model, 'model'):
            predictions = model.model(images)
        else:
            predictions = model(images)

        # Конвертируем для метрик
        for i in range(len(images)):
            # 🔥 Исправлено: вычитаем 1 из labels (Faster R-CNN: 0=фон, 1-5=классы)
            pred_labels = predictions[i]['labels'].cpu().numpy()
            if len(pred_labels) > 0:
                pred_labels = pred_labels - 1

            pred = {
                'boxes': predictions[i]['boxes'].cpu().numpy() if len(predictions[i]['boxes']) > 0 else np.array([]),
                'scores': predictions[i]['scores'].cpu().numpy() if len(predictions[i]['scores']) > 0 else np.array([]),
                'labels': pred_labels
            }

            # Берём target для этого изображения
            target = {
                'boxes': batch['boxes'][i].numpy() if len(batch['boxes'][i]) > 0 else np.array([]),
                'labels': batch['labels'][i].numpy() if len(batch['labels'][i]) > 0 else np.array([])
            }

            all_predictions.append(pred)
            all_targets.append(target)

    # Вычисляем метрики
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
    try:
        if hasattr(model, 'to'):
            model = model.to(device)
        elif hasattr(model, 'model') and hasattr(model.model, 'to'):
            model.model = model.model.to(device)
    except Exception as e:
        logger.warning(f"Could not move model to device: {e}")

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

def iou_single_faster_rcnn(box_p, box_t):
    x1_i = max(box_p[0], box_t[0])
    y1_i = max(box_p[1], box_t[1])
    x2_i = min(box_p[2], box_t[2])
    y2_i = min(box_p[3], box_t[3])
    inter = max(0.0, x2_i - x1_i) * max(0.0, y2_i - y1_i)
    area_p = (box_p[2] - box_p[0]) * (box_p[3] - box_p[1])
    area_t = (box_t[2] - box_t[0]) * (box_t[3] - box_t[1])
    union = area_p + area_t - inter
    return inter / (union + 1e-8)

def compute_precision_at_iou_epoch_faster_rcnn(all_preds, all_targets, iou_thresh=0.5):
    tp, fp = 0, 0

    for pred, target in zip(all_preds, all_targets):
        p_boxes = pred['boxes']
        p_labels = pred['labels']
        t_boxes = target['boxes']
        t_labels = target['labels']

        if p_boxes.numel() == 0:
            continue
        if t_boxes.numel() == 0:
            fp += p_boxes.size(0)
            continue

        used_target = torch.zeros(t_boxes.size(0), dtype=torch.bool)

        for pb, pl in zip(p_boxes, p_labels):
            best_iou = -1.0
            best_idx = -1

            for i, (tb, tl) in enumerate(zip(t_boxes, t_labels)):
                if used_target[i]:
                    continue
                if pl != tl:
                    continue

                iou = iou_single_faster_rcnn(pb, tb)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i

            if best_iou >= iou_thresh and best_idx != -1:
                tp += 1
                used_target[best_idx] = True
            else:
                fp += 1

    if tp + fp == 0:
        return 0.0
    return tp / (tp + fp)

def train_deformable_detr(
    model,
    train_loader,
    val_loader,
    epochs=50,
    lr=1e-4,
    device='cpu',
    save_dir='results/deformable_detr'
):
    """
    Обучение Deformable DETR.

    Args:
        model: Загруженная модель Deformable DETR (HuggingFace)
        train_loader: DataLoader для тренировки
        val_loader: DataLoader для валидации
        epochs: Количество эпох
        lr: Скорость обучения
        device: Устройство ('cpu' или 'cuda')
        save_dir: Папка для сохранения результатов
    """
    device = torch.device(device)
    model = model.to(device)

    # Информация о модели
    def print_model_info(model):
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print("\n" + "=" * 60)
        print("ИНФОРМАЦИЯ О МОДЕЛИ (Deformable DETR)")
        print("=" * 60)
        print(f"Всего параметров: {total_params:,} ({total_params / 1e6:.2f} M)")
        print(f"Обучаемых параметров: {trainable_params:,} ({trainable_params / 1e6:.2f} M)")

        if hasattr(model.config, 'model_type'):
            print(f"Тип модели: {model.config.model_type}")
        if hasattr(model.config, 'backbone'):
            print(f"Бэкбон: {model.config.backbone}")
        if hasattr(model.config, 'num_queries'):
            print(f"Количество запросов (queries): {model.config.num_queries}")
        if hasattr(model.config, 'd_model'):
            print(f"Размерность: {model.config.d_model}")
        if hasattr(model.config, 'encoder_layers'):
            print(f"Слоёв энкодера: {model.config.encoder_layers}")
        if hasattr(model.config, 'decoder_layers'):
            print(f"Слоёв декодера: {model.config.decoder_layers}")

        print("=" * 60 + "\n")

    print_model_info(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    history = {
        'train_loss': [],
        'val_loss': [],
        'mAP_0.5': [],
        'mAP_0.5_0.95': [],
        'precision': [],
        'recall': [],
        'f1': []
    }

    print(f"Training Deformable DETR on {device}")
    print(f"Epochs: {epochs}, LR: {lr}")
    print(f"Train: {len(train_loader.dataset)} images, Val: {len(val_loader.dataset)} images")

    for epoch in range(1, epochs + 1):
        # === TRAIN ===
        model.train()
        total_loss = 0

        for batch in tqdm(train_loader, desc=f'Epoch {epoch}/{epochs} [Train]'):
            pixel_values = batch['pixel_values'].to(device)

            targets = []
            for boxes, labels in zip(batch['boxes'], batch['labels']):
                boxes = boxes.to(device)
                labels = labels.to(device)

                if labels.numel() > 0:
                    targets.append({
                        'boxes': boxes,
                        'class_labels': labels + 1
                    })
                else:
                    targets.append({
                        'boxes': torch.zeros((0, 4), dtype=torch.float32, device=device),
                        'class_labels': torch.zeros((0,), dtype=torch.long, device=device)
                    })

            optimizer.zero_grad()
            outputs = model(pixel_values=pixel_values, labels=targets)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)

        # === VAL ===
        model.eval()
        val_loss = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f'Epoch {epoch}/{epochs} [Val]', leave=False):
                pixel_values = batch['pixel_values'].to(device)

                # Считаем loss
                targets = []
                for boxes, labels in zip(batch['boxes'], batch['labels']):
                    boxes = boxes.to(device)
                    labels = labels.to(device)
                    if labels.numel() > 0:
                        targets.append({'boxes': boxes, 'class_labels': labels + 1})
                    else:
                        targets.append({
                            'boxes': torch.zeros((0, 4), dtype=torch.float32, device=device),
                            'class_labels': torch.zeros((0,), dtype=torch.long, device=device)
                        })

                outputs_loss = model(pixel_values=pixel_values, labels=targets)
                val_loss += outputs_loss.loss.item()

                # Предсказания для mAP
                outputs = model(pixel_values=pixel_values, labels=None)

                logits = outputs['logits']
                pred_boxes = outputs['pred_boxes']
                probs = torch.nn.functional.softmax(logits, dim=-1)
                max_probs, pred_labels = probs.max(dim=-1)

                for i in range(pixel_values.shape[0]):
                    boxes = pred_boxes[i].cpu()
                    scores = max_probs[i].cpu()
                    labels = pred_labels[i].cpu()
                    mask = scores > 0.0001
                    all_preds.append({
                        'boxes': boxes[mask],
                        'scores': scores[mask],
                        'labels': labels[mask] - 1
                    })

                for boxes, labels in zip(batch['boxes'], batch['labels']):
                    if boxes.numel() > 0:
                        all_targets.append({
                            'boxes': boxes.cpu(),
                            'labels': labels.cpu()
                        })
                    else:
                        all_targets.append({
                            'boxes': torch.zeros((0, 4), dtype=torch.float32),
                            'labels': torch.zeros((0,), dtype=torch.long)
                        })

        avg_val_loss = val_loss / len(val_loader)
        history['val_loss'].append(avg_val_loss)

        # === МЕТРИКИ ===
        if all_preds and all_targets:
            metric = MeanAveragePrecision(iou_type='bbox')
            metric.update(all_preds, all_targets)
            map_metrics = metric.compute()

            map_50 = map_metrics['map_50'].item()
            map_50_95 = map_metrics['map'].item()
            precision = map_metrics.get('precision', torch.tensor(0.0)).mean().item()
            recall = map_metrics.get('recall', torch.tensor(0.0)).mean().item()
            f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

            history['mAP_0.5'].append(map_50)
            history['mAP_0.5_0.95'].append(map_50_95)
            history['precision'].append(precision)
            history['recall'].append(recall)
            history['f1'].append(f1)

            print(f"Epoch {epoch}/{epochs} | "
                  f"Train Loss: {avg_train_loss:.4f} | "
                  f"Val Loss: {avg_val_loss:.4f} | "
                  f"mAP@0.5: {map_50:.4f} | "
                  f"P: {precision:.4f} | R: {recall:.4f} | F1: {f1:.4f}")
        else:
            print(f"Epoch {epoch}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | ⚠️ Нет данных")
            history['mAP_0.5'].append(0.0)
            history['mAP_0.5_0.95'].append(0.0)
            history['precision'].append(0.0)
            history['recall'].append(0.0)
            history['f1'].append(0.0)

    # Сохраняем модель и историю
    model.save_pretrained(save_dir)
    torch.save(model.state_dict(), save_dir / 'model.pt')

    with open(save_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print(f'Обучение Deformable DETR завершено! Результаты сохранены в {save_dir}')

    try:
        plot_training_history(
            history_path=str(save_dir / 'history.json'),
            save_dir='results/plots',
            model_name='Deformable-DETR'
        )

        logger.info("Графики Faster R-CNN сохранены в results/plots/")
    except Exception as e:
        logger.warning(f"Не удалось построить графики: {e}")

    return history

def train_faster_rcnn(
        model,
        train_loader,
        val_loader,
        epochs=100,
        lr=1e-4,
        weight_decay=1e-4,
        device='cpu',
        save_dir='results/faster_rcnn'
):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(device)
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = {
        'train_loss': [],
        'val_precision': [],
        'learning_rate': [],
        'mAP_0.5': [],
        'mAP_0.5_0.95': [],
        'precision': [],
        'recall': []
    }

    logger.info(f"Training Faster R-CNN on {device}")
    logger.info(f"Epochs: {epochs}, LR: {lr}")
    logger.info(f"Train: {len(train_loader.dataset)} images, Val: {len(val_loader.dataset)} images")

    best_val_prec = -1.0

    for epoch in range(1, epochs + 1):
        # === TRAIN ===
        model.train()
        train_loss = 0.0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]")

        for batch in train_bar:
            pixel_values = batch['pixel_values'].to(device)
            targets_raw = batch['targets']

            targets = []
            for t in targets_raw:
                targets.append({
                    'boxes': t['boxes'].to(device),
                    'labels': t['labels'].to(device)
                })

            optimizer.zero_grad()
            outputs = model(pixel_values, targets)

            if not isinstance(outputs, dict):
                raise RuntimeError("Training: model did not return a loss dict.")

            loss = sum(loss_val for loss_val in outputs.values())
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_train_loss = train_loss / max(1, len(train_loader))
        history['train_loss'].append(avg_train_loss)

        # === VAL (с mAP) ===
        model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [Val]", leave=False)
            for batch in val_bar:
                pixel_values = batch['pixel_values'].to(device)
                targets_raw = batch['targets']

                outputs = model(pixel_values)

                # Форматируем предсказания для torchmetrics
                for out in outputs:
                    boxes = out['boxes'].cpu()
                    scores = out['scores'].cpu()
                    labels = out['labels'].cpu()
                    all_preds.append({
                        'boxes': boxes,
                        'scores': scores,
                        'labels': labels
                    })

                # Форматируем цели
                for t in targets_raw:
                    all_targets.append({
                        'boxes': t['boxes'].cpu(),
                        'labels': t['labels'].cpu()
                    })

        # Считаем mAP
        metric = MeanAveragePrecision(iou_type='bbox')
        metric.update(all_preds, all_targets)
        map_metrics = metric.compute()

        # Извлекаем нужные метрики
        map_50 = map_metrics['map_50'].item()
        map_50_95 = map_metrics['map'].item()
        precision = map_metrics.get('precision', torch.tensor(0.0)).mean().item()
        recall = map_metrics.get('recall', torch.tensor(0.0)).mean().item()

        # Сохраняем в history
        history['mAP_0.5'].append(map_50)
        history['mAP_0.5_0.95'].append(map_50_95)
        history['precision'].append(precision)
        history['recall'].append(recall)

        val_prec = compute_precision_at_iou_epoch_faster_rcnn(all_preds, all_targets, iou_thresh=0.5)
        history['val_precision'].append(val_prec)
        history['learning_rate'].append(optimizer.param_groups[0]['lr'])

        logger.info(f"Epoch {epoch:3d}/{epochs} | Loss: {avg_train_loss:.4f} | mAP@0.5: {map_50:.4f} | mAP@0.5:0.95: {map_50_95:.4f} | P: {precision:.4f} | R: {recall:.4f}")
        # Сохраняем лучшую модель по precision
        if val_prec > best_val_prec:
            best_val_prec = val_prec
            best_checkpoint_path = save_dir / 'faster_rcnn_best.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_precision': val_prec,
                'history': history
            }, best_checkpoint_path)
            logger.info(f"New best model saved: {best_checkpoint_path} (Prec@0.5={val_prec:.4f})")

        # Чекпоинт каждые 10 эпох
        if epoch % 10 == 0:
            checkpoint_path = save_dir / f'faster_rcnn_epoch_{epoch}.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history': history
            }, checkpoint_path)
            logger.info(f"Checkpoint: {checkpoint_path}")

    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'history': history
    }, save_dir / 'faster_rcnn_final.pt')

    with open(save_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)

    best_prec = max(history['val_precision'])
    logger.info(f"Training complete! Best val Prec@0.5: {best_prec:.4f}")

    try:
        plot_training_history(
            history_path=str(save_dir / 'history.json'),
            save_dir='results/plots',
            model_name='Faster-RCNN'
        )
        logger.info("Графики Faster R-CNN сохранены в results/plots/")
    except Exception as e:
        logger.warning(f"Не удалось построить графики: {e}")

    return history

def train_rtdetr(
        model,
        train_loader,
        val_loader,
        epochs=10,
        lr=1e-4,
        device='cpu',
        save_dir='results/rtdetr'
):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(device)
    model = model.to(device)

    # RT-DETR через ultralytics использует свой пайплайн
    # Но мы можем использовать стандартный подход с PyTorch

    # Создаём data.yaml для RT-DETR с абсолютным путём
    data_dir = Path(train_loader.dataset.data_dir).absolute()
    data_yaml = {
        'path': str(data_dir),  # <-- ТЕПЕРЬ ПРАВИЛЬНО: абсолютный путь к папке, где лежат train/ и val/
        'train': 'train/images',
        'val': 'val/images',
        'nc': 5,
        'names': ['creeper', 'skeleton', 'spider', 'zombie', 'enderman']
    }

    data_yaml_path = save_dir / 'data.yaml'
    with open(data_yaml_path, 'w') as f:
        yaml.dump(data_yaml, f)

    # Используем встроенное обучение RT-DETR
    results = model.model.train(
        data=str(data_yaml_path),
        epochs=epochs,
        imgsz=320,
        batch=4,
        lr0=lr,
        device='cpu',
        project=str(save_dir),
        name='rtdetr_run',
        exist_ok=True,
        verbose=True,
        plots=True
    )

    # Собираем историю из результатов
    history = {
        'train_loss': [],
        'val_loss': [],
        'learning_rate': [lr],
        'mAP_0.5': [],
        'mAP_0.5_0.95': [],
        'precision': [],
        'recall': []
    }

    # Извлекаем метрики из результатов (если доступны)
    if hasattr(results, 'metrics'):
        history['mAP_0.5'].append(results.metrics.get('mAP_0.5', 0.0))
        history['mAP_0.5_0.95'].append(results.metrics.get('mAP_0.5_0.95', 0.0))
        history['precision'].append(results.metrics.get('precision', 0.0))
        history['recall'].append(results.metrics.get('recall', 0.0))

    logger.info(f"Training complete! Best mAP@0.5: {max(history['mAP_0.5']) if history['mAP_0.5'] else 0.0:.4f}")
    return history

def train_model_yolo_world(
    model,
    data_yaml_path: str,
    config: Dict[str, Any],
    model_name: str = 'yolo_world'
) -> Dict[str, Any]:
    """
    Запускает обучение модели через Ultralytics.

    Args:
        model: Модель YOLO / YOLO-World
        data_yaml_path: Путь к data.yaml
        config: Конфигурация модели из default.yaml
        model_name: Имя модели для сохранения

    Returns:
        Dict: Результаты обучения
    """
    results = model.train(
        data=data_yaml_path,
        epochs=config['epochs'],
        imgsz=config.get('image_size', 640),
        batch=config.get('batch_size', 16),
        lr0=config.get('lr', 0.001),
        device=config.get('device', 'cpu'),
        project='results/logs',
        name=model_name,
        exist_ok=True,
        verbose=True
    )

    return {
        'model_path': results.save_dir,
        'results': results
    }