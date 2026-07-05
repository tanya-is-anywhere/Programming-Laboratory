import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import yaml
import json
import torch
from datetime import datetime
import seaborn as sns
from PIL import Image
import cv2
import random

logger = logging.getLogger(__name__)


# ============================================
# Настройка и логирование
# ============================================

def setup_logging(log_dir: Path, verbose: bool = False) -> None:
    """
    Настройка логирования в файл и консоль

    Args:
        log_dir: Директория для логов
        verbose: Подробный вывод (DEBUG уровень)
    """
    log_dir.mkdir(parents=True, exist_ok=True)

    level = logging.DEBUG if verbose else logging.INFO

    # Форматтер с цветами для разных уровней (опционально)
    class ColoredFormatter(logging.Formatter):
        """Форматтер с цветами для консоли"""
        COLORS = {
            'DEBUG': '\033[94m',  # Синий
            'INFO': '\033[92m',  # Зеленый
            'WARNING': '\033[93m',  # Желтый
            'ERROR': '\033[91m',  # Красный
            'CRITICAL': '\033[91m\033[1m',  # Жирный красный
        }
        RESET = '\033[0m'

        def format(self, record):
            color = self.COLORS.get(record.levelname, self.RESET)
            record.levelname = f"{color}{record.levelname}{self.RESET}"
            return super().format(record)

    # Файловый handler (без цветов)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler = logging.FileHandler(log_dir / 'experiment.log', encoding='utf-8')
    file_handler.setLevel(level)
    file_handler.setFormatter(file_formatter)

    # Консольный handler (с цветами)
    console_formatter = ColoredFormatter(
        '%(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO if not verbose else logging.DEBUG)
    console_handler.setFormatter(console_formatter)

    # Настройка корневого логгера
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers.clear()  # Очищаем предыдущие хендлеры
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    logger.info(f"Logging configured. Logs will be saved to {log_dir}")


def save_config(config: Dict, path: Path) -> None:
    """
    Сохранение конфигурации в YAML файл

    Args:
        config: Словарь с конфигурацией
        path: Путь для сохранения
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    logger.info(f"Config saved to {path}")


# ============================================
# Визуализация обучения
# ============================================

def plot_training_history(
        history: Dict[str, List[float]],
        model_name: str,
        config: Union[Dict, str],
        save_dir: Optional[Path] = None
) -> None:
    """
    Строит комплексные графики обучения

    Args:
        history: История обучения с ключами:
            - train_loss, val_loss
            - train_cls_loss, val_cls_loss
            - train_box_loss, val_box_loss
            - mAP_50, mAP_50_95 (по эпохам)
            - precision, recall
            - learning_rate
        model_name: Название модели
        config: Конфигурация или путь к директории
        save_dir: Альтернативная директория для сохранения
    """
    # Определяем директорию для сохранения
    if save_dir:
        plots_dir = Path(save_dir)
    elif isinstance(config, dict):
        plots_dir = Path(config.get('results', {}).get('plots_dir', 'results/plots'))
    else:
        plots_dir = Path(config)

    plots_dir = Path(plots_dir) / model_name
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Стиль графиков
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")

    # 1. График Loss
    if 'train_loss' in history and 'val_loss' in history:
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # Total Loss
        ax = axes[0]
        ax.plot(history['train_loss'], label='Train Loss', linewidth=2, marker='o', markersize=3)
        ax.plot(history['val_loss'], label='Val Loss', linewidth=2, marker='s', markersize=3)
        ax.set_title(f'{model_name.upper()} - Total Loss', fontsize=14, fontweight='bold')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        # Добавляем аннотацию минимального loss
        min_val_loss_epoch = np.argmin(history['val_loss'])
        min_val_loss = history['val_loss'][min_val_loss_epoch]
        ax.annotate(f'Best: {min_val_loss:.4f}',
                    xy=(min_val_loss_epoch, min_val_loss),
                    xytext=(10, 20), textcoords='offset points',
                    arrowprops=dict(arrowstyle='->', color='red'),
                    fontsize=10, color='red')

        # Component Losses (если есть)
        ax = axes[1]
        loss_components = ['train_cls_loss', 'train_box_loss', 'train_dfl_loss']
        for comp in loss_components:
            if comp in history:
                ax.plot(history[comp], label=comp.replace('train_', '').replace('_', ' ').title(),
                        linewidth=1.5, alpha=0.7)
        ax.set_title(f'{model_name.upper()} - Loss Components', fontsize=14, fontweight='bold')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(plots_dir / f'{model_name}_loss.png', dpi=150, bbox_inches='tight')
        plt.close()
        logger.debug(f"Loss plot saved for {model_name}")

    # 2. График метрик (mAP, Precision, Recall)
    metric_keys = ['mAP_50', 'mAP_50_95', 'precision', 'recall', 'f1_score']
    available_metrics = [k for k in metric_keys if k in history and len(history[k]) > 0]

    if available_metrics:
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # mAP
        ax = axes[0]
        for metric in ['mAP_50', 'mAP_50_95']:
            if metric in history:
                ax.plot(history[metric], label=metric.replace('_', '@'),
                        linewidth=2, marker='o', markersize=3)
        ax.set_title(f'{model_name.upper()} - mAP', fontsize=14, fontweight='bold')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('mAP', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])

        # Аннотация лучшего mAP
        if 'mAP_50' in history:
            best_map_epoch = np.argmax(history['mAP_50'])
            best_map = history['mAP_50'][best_map_epoch]
            ax.annotate(f'Best mAP@50: {best_map:.4f}',
                        xy=(best_map_epoch, best_map),
                        xytext=(10, -20), textcoords='offset points',
                        arrowprops=dict(arrowstyle='->', color='green'),
                        fontsize=10, color='green')

        # Precision/Recall/F1
        ax = axes[1]
        for metric in ['precision', 'recall', 'f1_score']:
            if metric in history:
                ax.plot(history[metric], label=metric.replace('_', ' ').title(),
                        linewidth=2, marker='s', markersize=3)
        ax.set_title(f'{model_name.upper()} - Precision & Recall', fontsize=14, fontweight='bold')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])

        plt.tight_layout()
        plt.savefig(plots_dir / f'{model_name}_metrics.png', dpi=150, bbox_inches='tight')
        plt.close()
        logger.debug(f"Metrics plot saved for {model_name}")

    # 3. График Learning Rate (если есть)
    if 'learning_rate' in history and len(history['learning_rate']) > 0:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(history['learning_rate'], linewidth=2, color='purple')
        ax.set_title(f'{model_name.upper()} - Learning Rate Schedule', fontsize=14, fontweight='bold')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Learning Rate', fontsize=12)
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(plots_dir / f'{model_name}_lr.png', dpi=150, bbox_inches='tight')
        plt.close()
        logger.debug(f"Learning rate plot saved for {model_name}")

    # 4. Сводный дашборд (все метрики на одном графике)
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Loss
    if 'train_loss' in history:
        axes[0, 0].plot(history['train_loss'], label='Train', linewidth=1.5)
        axes[0, 0].plot(history['val_loss'], label='Val', linewidth=1.5)
        axes[0, 0].set_title('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

    # mAP
    if 'mAP_50' in history:
        axes[0, 1].plot(history['mAP_50'], label='mAP@50', linewidth=1.5)
        if 'mAP_50_95' in history:
            axes[0, 1].plot(history['mAP_50_95'], label='mAP@50:95', linewidth=1.5)
        axes[0, 1].set_title('mAP')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

    # Precision/Recall
    if 'precision' in history:
        axes[1, 0].plot(history['precision'], label='Precision', linewidth=1.5)
        if 'recall' in history:
            axes[1, 0].plot(history['recall'], label='Recall', linewidth=1.5)
        axes[1, 0].set_title('Precision & Recall')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

    # LR
    if 'learning_rate' in history:
        axes[1, 1].plot(history['learning_rate'], linewidth=1.5, color='orange')
        axes[1, 1].set_title('Learning Rate')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle(f'{model_name.upper()} - Training Dashboard', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(plots_dir / f'{model_name}_dashboard.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Training plots saved to {plots_dir}")


# ============================================
# Визуализация предсказаний
# ============================================

def visualize_predictions(
        model: Any,
        dataloader: torch.utils.data.DataLoader,
        model_name: str,
        config: Union[Dict, str],
        num_images: int = 5,
        save_dir: Optional[Path] = None,
        class_names: Optional[List[str]] = None,
        conf_threshold: float = 0.25,
        show_ground_truth: bool = True
) -> None:
    """
    Визуализирует предсказания модели с боксами

    Args:
        model: Модель детекции (должна иметь метод predict)
        dataloader: DataLoader с изображениями
        model_name: Название модели
        config: Конфигурация
        num_images: Количество изображений для визуализации
        save_dir: Директория для сохранения
        class_names: Список имен классов
        conf_threshold: Порог уверенности
        show_ground_truth: Показывать ли ground truth боксы
    """
    # Определяем директорию
    if save_dir:
        pred_dir = Path(save_dir)
    elif isinstance(config, dict):
        pred_dir = Path(config.get('results', {}).get('plots_dir', 'results/plots'))
    else:
        pred_dir = Path(config)

    pred_dir = Path(pred_dir) / model_name / 'predictions'
    pred_dir.mkdir(parents=True, exist_ok=True)

    # Цвета для классов
    colors = plt.cm.tab20(np.linspace(0, 1, 20))
    if class_names:
        class_colors = {name: colors[i % len(colors)] for i, name in enumerate(class_names)}
    else:
        class_colors = {}

    model.eval()
    device = next(model.model.parameters()).device if hasattr(model, 'model') else 'cpu'

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= num_images:
                break

            # Получаем изображения из батча
            if isinstance(batch, dict):
                images = batch['image']
                targets = batch.get('target', None)
            elif isinstance(batch, (list, tuple)):
                images = batch[0]
                targets = batch[1] if len(batch) > 1 else None
            else:
                images = batch

            # Берем первое изображение из батча
            if isinstance(images, torch.Tensor):
                image = images[0]
            else:
                image = images[0]

            # Конвертируем тензор в numpy для отображения
            if isinstance(image, torch.Tensor):
                # Предполагаем формат CxHxW с нормализацией
                if image.shape[0] == 3:
                    # Денормализация (ImageNet stats)
                    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                    image = image * std + mean
                    image = image.clamp(0, 1)
                image_np = image.permute(1, 2, 0).cpu().numpy()
            else:
                image_np = image

            # Создаем фигуру
            fig, axes = plt.subplots(1, 2 if show_ground_truth and targets is not None else 1,
                                     figsize=(15 if show_ground_truth else 8, 8))

            if not isinstance(axes, np.ndarray):
                axes = [axes]

            # Предсказание модели
            try:
                prediction = model.predict(image_np, conf_threshold=conf_threshold)
            except Exception as e:
                logger.warning(f"Prediction failed: {e}")
                prediction = {'boxes': [], 'scores': [], 'labels': [], 'class_names': []}

            # ---- Ground Truth (если есть) ----
            if show_ground_truth and targets is not None:
                ax = axes[0]
                ax.imshow(image_np)
                ax.set_title('Ground Truth', fontsize=14, fontweight='bold')

                if isinstance(targets, dict):
                    gt_boxes = targets['boxes'][0] if len(targets['boxes']) > 0 else []
                    gt_labels = targets['labels'][0] if len(targets['labels']) > 0 else []
                elif isinstance(targets, torch.Tensor):
                    gt_boxes = targets[0]['boxes'] if len(targets) > 0 else []
                    gt_labels = targets[0]['labels'] if len(targets) > 0 else []
                else:
                    gt_boxes = []
                    gt_labels = []

                # Рисуем ground truth боксы
                for box, label in zip(gt_boxes, gt_labels):
                    if isinstance(label, torch.Tensor):
                        label = label.item()

                    x1, y1, x2, y2 = box[:4].cpu().numpy() if isinstance(box, torch.Tensor) else box[:4]
                    class_name = class_names[label] if class_names and label < len(class_names) else f'Class {label}'
                    color = class_colors.get(class_name, 'red')

                    rect = patches.Rectangle(
                        (x1, y1), x2 - x1, y2 - y1,
                        linewidth=2, edgecolor=color, facecolor='none'
                    )
                    ax.add_patch(rect)
                    ax.text(x1, y1 - 5, class_name,
                            fontsize=8, color='white',
                            bbox=dict(boxstyle='round', facecolor=color, alpha=0.8))

                ax.axis('off')

            # ---- Предсказания модели ----
            ax = axes[1] if show_ground_truth and targets is not None else axes[0]
            ax.imshow(image_np)
            ax.set_title(f'{model_name.upper()} Predictions', fontsize=14, fontweight='bold')

            # Рисуем предсказанные боксы
            for box, score, class_name in zip(
                    prediction['boxes'],
                    prediction['scores'],
                    prediction['class_names']
            ):
                if score < conf_threshold:
                    continue

                x1, y1, x2, y2 = box[:4]
                color = class_colors.get(class_name, 'blue')

                rect = patches.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    linewidth=2, edgecolor=color, facecolor='none'
                )
                ax.add_patch(rect)

                label_text = f'{class_name}: {score:.2f}'
                ax.text(x1, y1 - 10, label_text,
                        fontsize=8, color='white',
                        bbox=dict(boxstyle='round', facecolor=color, alpha=0.8))

            # Если нет детекций
            if len(prediction['boxes']) == 0:
                ax.text(0.5, 0.5, 'No detections',
                        ha='center', va='center',
                        transform=ax.transAxes,
                        fontsize=14, color='red',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            ax.axis('off')

            # Сохраняем
            plt.tight_layout()
            plt.savefig(pred_dir / f'prediction_{batch_idx}.png', dpi=150, bbox_inches='tight')
            plt.close()

    logger.info(f"Prediction visualizations saved to {pred_dir}")


def visualize_comparison(
        models_predictions: Dict[str, List[Dict]],
        image: np.ndarray,
        save_path: Path,
        class_names: Optional[List[str]] = None
) -> None:
    """
    Визуализация сравнения предсказаний разных моделей на одном изображении

    Args:
        models_predictions: Словарь {имя_модели: список_предсказаний}
        image: Исходное изображение
        save_path: Путь для сохранения
        class_names: Список имен классов
    """
    num_models = len(models_predictions)
    fig, axes = plt.subplots(1, num_models + 1, figsize=(5 * (num_models + 1), 5))

    if not isinstance(axes, np.ndarray):
        axes = [axes]

    # Оригинальное изображение
    axes[0].imshow(image)
    axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0].axis('off')

    # Предсказания каждой модели
    colors = plt.cm.tab10(np.linspace(0, 1, num_models))

    for idx, (model_name, predictions) in enumerate(models_predictions.items()):
        ax = axes[idx + 1]
        ax.imshow(image)
        ax.set_title(model_name.upper(), fontsize=12, fontweight='bold')

        for pred in predictions:
            for box, score, class_name in zip(
                    pred.get('boxes', []),
                    pred.get('scores', []),
                    pred.get('class_names', [])
            ):
                x1, y1, x2, y2 = box[:4]
                color = colors[idx]

                rect = patches.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    linewidth=2, edgecolor=color, facecolor='none'
                )
                ax.add_patch(rect)
                ax.text(x1, y1 - 5, f'{class_name}: {score:.2f}',
                        fontsize=7, color='white',
                        bbox=dict(boxstyle='round', facecolor=color, alpha=0.8))

        ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Comparison visualization saved to {save_path}")


# ============================================
# Сравнительный анализ
# ============================================

def create_comparison_table(metrics: Dict[str, Dict]) -> str:
    """
    Создание красивой таблицы сравнения моделей

    Args:
        metrics: Словарь с метриками для каждой модели

    Returns:
        Строка с форматированной таблицей
    """
    # Пытаемся использовать rich для красивых таблиц
    try:
        from rich.console import Console
        from rich.table import Table

        table = Table(title="🎮 Minecraft Mobs Detection - Model Comparison",
                      title_style="bold cyan")

        table.add_column("Model", style="cyan", no_wrap=True)
        table.add_column("mAP@0.5", justify="right", style="green")
        table.add_column("mAP@0.5:0.95", justify="right", style="green")
        table.add_column("Precision", justify="right")
        table.add_column("Recall", justify="right")
        table.add_column("F1-Score", justify="right")
        table.add_column("Inference (ms)", justify="right", style="yellow")
        table.add_column("Params (M)", justify="right", style="yellow")

        for model_name, m in metrics.items():
            table.add_row(
                model_name.upper(),
                f"{m.get('mAP_50', 0):.4f}",
                f"{m.get('mAP_50_95', 0):.4f}",
                f"{m.get('precision', 0):.4f}",
                f"{m.get('recall', 0):.4f}",
                f"{m.get('f1_score', 0):.4f}",
                f"{m.get('inference_time', 0):.1f}",
                f"{m.get('parameters', 0) / 1e6:.1f}"
            )

        # Выделяем лучшие результаты
        # ... можно добавить логику для выделения жирным

        console = Console()
        with console.capture() as capture:
            console.print(table)
        return capture.get()

    except ImportError:
        # Fallback без rich
        lines = ["\n" + "=" * 90]
        lines.append("🎮 MINECRAFT MOBS DETECTION - MODEL COMPARISON")
        lines.append("=" * 90)
        lines.append(
            f"{'Model':<15} {'mAP@0.5':<10} {'mAP@.5:.95':<12} {'Precision':<10} {'Recall':<10} {'F1':<10} {'Time(ms)':<10} {'Params(M)':<10}")
        lines.append("-" * 90)

        for model_name, m in metrics.items():
            lines.append(
                f"{model_name.upper():<15} "
                f"{m.get('mAP_50', 0):<10.4f} "
                f"{m.get('mAP_50_95', 0):<12.4f} "
                f"{m.get('precision', 0):<10.4f} "
                f"{m.get('recall', 0):<10.4f} "
                f"{m.get('f1_score', 0):<10.4f} "
                f"{m.get('inference_time', 0):<10.1f} "
                f"{m.get('parameters', 0) / 1e6:<10.1f}"
            )

        lines.append("=" * 90)
        return '\n'.join(lines)


def plot_comparative_analysis(
        all_metrics: Dict[str, Dict],
        save_dir: Path,
        metric_keys: List[str] = ['mAP_50', 'mAP_50_95', 'precision', 'recall', 'f1_score']
) -> None:
    """
    Строит сравнительные графики для всех моделей

    Args:
        all_metrics: Словарь с метриками моделей
        save_dir: Директория для сохранения
        metric_keys: Ключи метрик для визуализации
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    model_names = list(all_metrics.keys())

    # Групповой bar chart
    fig, axes = plt.subplots(1, len(metric_keys), figsize=(5 * len(metric_keys), 6))

    if len(metric_keys) == 1:
        axes = [axes]

    for idx, metric in enumerate(metric_keys):
        values = [all_metrics[m].get(metric, 0) for m in model_names]

        # Сортируем по убыванию
        sorted_indices = np.argsort(values)[::-1]
        sorted_models = [model_names[i] for i in sorted_indices]
        sorted_values = [values[i] for i in sorted_indices]

        colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(model_names)))
        bars = axes[idx].bar(sorted_models, sorted_values, color=colors)

        # Добавляем значения над барами
        for bar, val in zip(bars, sorted_values):
            axes[idx].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                           f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

        axes[idx].set_title(metric.replace('_', '@').upper(), fontsize=14, fontweight='bold')
        axes[idx].set_ylabel(metric, fontsize=12)
        axes[idx].tick_params(axis='x', rotation=45)
        axes[idx].grid(axis='y', alpha=0.3)
        axes[idx].set_ylim([0, 1.0])

    plt.suptitle('Model Comparison - Minecraft Mobs Detection', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_dir / 'model_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Радарная диаграмма
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

    angles = np.linspace(0, 2 * np.pi, len(metric_keys), endpoint=False).tolist()
    angles += angles[:1]  # Замыкаем круг

    for idx, model_name in enumerate(model_names):
        values = [all_metrics[model_name].get(m, 0) for m in metric_keys]
        values += values[:1]  # Замыкаем

        ax.plot(angles, values, 'o-', linewidth=2, label=model_name.upper(), markersize=6)
        ax.fill(angles, values, alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m.replace('_', '@').upper() for m in metric_keys], fontsize=10)
    ax.set_ylim([0, 1])
    ax.set_title('Model Comparison - Radar Chart', fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(save_dir / 'radar_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Comparative analysis plots saved to {save_dir}")


# ============================================
# Вспомогательные функции
# ============================================

def set_seed(seed: int = 42) -> None:
    """
    Установка random seed для воспроизводимости

    Args:
        seed: Значение seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Random seed set to {seed}")


def get_device() -> torch.device:
    """
    Определение доступного устройства

    Returns:
        torch.device
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        logger.info("Using MPS (Apple Silicon)")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU")

    return device


def save_json(data: Dict, path: Path) -> None:
    """Сохранение данных в JSON"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.debug(f"JSON saved to {path}")


def load_json(path: Path) -> Dict:
    """Загрузка данных из JSON"""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def format_time(seconds: float) -> str:
    """Форматирование времени"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"
