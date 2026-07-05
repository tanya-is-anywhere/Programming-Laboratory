"""
Модуль для обучения моделей детекции
"""

from .train import (
    train_model,
    train_one_epoch,
    validate,
    EarlyStopping,
    ModelCheckpoint,
    create_optimizer,
    create_scheduler
)

__all__ = [
    'train_model',
    'train_one_epoch',
    'validate',
    'EarlyStopping',
    'ModelCheckpoint',
    'create_optimizer',
    'create_scheduler'
]
