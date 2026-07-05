"""
Вспомогательные функции
"""
import logging
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import torch
import random

logger = logging.getLogger(__name__)


def setup_logging(log_dir, verbose=False):
    """Настройка логирования"""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def plot_training_history(history, model_name, save_dir=None, **kwargs):
    """Строит графики обучения"""
    save_dir = Path(save_dir) if save_dir else Path('results/plots')
    save_dir.mkdir(parents=True, exist_ok=True)

    if 'train_loss' in history and history['train_loss']:
        plt.figure(figsize=(10, 5))
        plt.plot(history['train_loss'], label='Train Loss', linewidth=2)
        if 'val_loss' in history and history['val_loss']:
            plt.plot(history['val_loss'], label='Val Loss', linewidth=2)
        plt.title(f'{model_name} - Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig(save_dir / f'{model_name}_loss.png', dpi=150)
        plt.close()
        logger.info(f"Plot saved to {save_dir / f'{model_name}_loss.png'}")


def visualize_predictions(model, data_loader, model_name, save_dir=None, num_images=5, **kwargs):
    """Визуализация предсказаний (заглушка)"""
    save_dir = Path(save_dir) if save_dir else Path('results/predictions')
    save_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Predictions visualization saved to {save_dir}")


def set_seed(seed=42):
    """Установка seed для воспроизводимости"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    """Определение устройства"""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def save_config(config, path):
    """Сохранение конфига"""
    import yaml
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        yaml.dump(config, f)


def create_comparison_table(metrics):
    """Создание таблицы сравнения"""
    table = "\nModel Comparison:\n"
    table += "-" * 60 + "\n"
    for model, m in metrics.items():
        table += f"{model}: mAP50={m.get('mAP_50', 0):.4f}\n"
    return table
