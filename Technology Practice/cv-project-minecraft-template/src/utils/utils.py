"""
Вспомогательные функции для проекта
"""

import json
import logging
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import yaml

logger = logging.getLogger(__name__)


# ============================================
# 1. ЛОГИРОВАНИЕ
# ============================================

def setup_file_logging(log_dir: str = 'results/logs', experiment_name: str = None) -> str:
    """
    Настраивает логирование в файл с именем experiment_YYYYMMDD_HHMMSS.log

    Args:
        log_dir: Папка для сохранения логов
        experiment_name: Имя эксперимента (если None — создаётся автоматически)

    Returns:
        str: Путь к созданному лог-файлу
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    if experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"experiment_{timestamp}"

    log_file = log_dir / f"{experiment_name}.log"

    # Настраиваем корневой логгер
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Удаляем существующие обработчики (чтобы не дублировать)
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Создаём форматтер
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Обработчик для файла
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    # Обработчик для консоли
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    logger.info(f"Эксперимент запущен: {experiment_name}")
    logger.info(f"Лог-файл: {log_file.absolute()}")
    logger.info(f"Время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return str(log_file)


# ============================================
# 2. КОНФИГУРАЦИЯ
# ============================================

def load_config(config_path: str) -> dict:
    """Загружает YAML-конфиг из файла."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Конфиг не найден: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    return config


def save_config(config: dict, path: Union[str, Path]) -> None:
    """Сохраняет словарь в YAML-файл."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    logger.info(f"Конфиг сохранён: {path}")


# ============================================
# 3. ВОСПРОИЗВОДИМОСТЬ
# ============================================

def set_seed(seed: int = 42) -> None:
    """Установка random seed для воспроизводимости."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Seed установлен: {seed}")


def get_device() -> torch.device:
    """Определение доступного устройства."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"Используется GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        logger.info("Используется CPU")
    return device


# ============================================
# 4. ВИЗУАЛИЗАЦИЯ ОБУЧЕНИЯ
# ============================================

def plot_training_history(
    history: Dict[str, List[float]],
    model_name: str,
    save_dir: Optional[Path] = None
) -> None:
    """
    Строит графики обучения: loss и метрики.

    Args:
        history: Словарь с историями (train_loss, val_loss, mAP_0.5, precision, recall)
        model_name: Имя модели для подписи
        save_dir: Папка для сохранения
    """
    if save_dir is None:
        save_dir = Path('results/plots')
    save_dir = save_dir / model_name
    save_dir.mkdir(parents=True, exist_ok=True)

    plt.style.use('seaborn-v0_8-darkgrid')

    # 1. График Loss
    if 'train_loss' in history and history['train_loss']:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(history['train_loss'], label='Train Loss', linewidth=2, marker='o', markersize=3)

        if 'val_loss' in history and history['val_loss']:
            ax.plot(history['val_loss'], label='Val Loss', linewidth=2, marker='s', markersize=3)

        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title(f'{model_name.upper()} - Training Loss', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_dir / f'{model_name}_loss.png', dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"График loss сохранён: {save_dir / f'{model_name}_loss.png'}")

    # 2. График метрик
    metrics = ['mAP_0.5', 'mAP_0.5_0.95', 'precision', 'recall']
    available = [m for m in metrics if m in history and history[m]]

    if available:
        fig, ax = plt.subplots(figsize=(10, 5))

        for metric in available:
            label = metric.replace('_', '@') if 'mAP' in metric else metric.capitalize()
            ax.plot(history[metric], label=label, linewidth=2, marker='o', markersize=3)

        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title(f'{model_name.upper()} - Metrics', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])

        plt.tight_layout()
        plt.savefig(save_dir / f'{model_name}_metrics.png', dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"График метрик сохранён: {save_dir / f'{model_name}_metrics.png'}")


# ============================================
# 5. ТАБЛИЦА СРАВНЕНИЯ
# ============================================

def create_comparison_table(metrics: Dict[str, Dict]) -> str:
    """
    Создаёт текстовую таблицу сравнения моделей.

    Args:
        metrics: Словарь {model_name: {mAP_0.5, mAP_0.5_0.95, precision, recall, ...}}

    Returns:
        str: Отформатированная таблица
    """
    lines = []
    lines.append("\n" + "=" * 90)
    lines.append("MINECRAFT MOBS DETECTION - MODEL COMPARISON")
    lines.append("=" * 90)

    header = (
        f"{'Model':<15} "
        f"{'mAP@0.5':<12} "
        f"{'mAP@0.5:0.95':<15} "
        f"{'Precision':<12} "
        f"{'Recall':<10} "
        f"{'F1':<10} "
    )
    lines.append(header)
    lines.append("-" * 90)

    for model_name, m in metrics.items():
        f1 = m.get('f1_score', 0)
        if f1 == 0 and 'precision' in m and 'recall' in m:
            p = m.get('precision', 0)
            r = m.get('recall', 0)
            f1 = 2 * (p * r) / (p + r + 1e-8)

        line = (
            f"{model_name.upper():<15} "
            f"{m.get('mAP_0.5', 0):<12.4f} "
            f"{m.get('mAP_0.5_0.95', 0):<15.4f} "
            f"{m.get('precision', 0):<12.4f} "
            f"{m.get('recall', 0):<10.4f} "
            f"{f1:<10.4f} "
        )
        lines.append(line)

    lines.append("=" * 90)
    return '\n'.join(lines)


# ============================================
# 6. JSON
# ============================================

def save_json(data: Dict, path: Union[str, Path]) -> None:
    """Сохраняет словарь в JSON-файл."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    logger.debug(f"JSON сохранён: {path}")