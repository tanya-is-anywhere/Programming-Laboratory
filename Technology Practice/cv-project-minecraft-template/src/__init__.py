"""
Minecraft Mobs Detection Package
"""

__version__ = "1.0.0"
__author__ = "Shilintseva Tatyana V."

import logging

logger = logging.getLogger(__name__)

# Импорт подпакетов
from . import dataset
from . import models
from . import training
from . import evaluation
from . import utils
from . import inference

# Основные импорты для удобства
from .dataset import get_dataloader, MinecraftMobsDataset
from .models import get_model, MODEL_REGISTRY
from .training import train_model
from .evaluation import evaluate_model, save_metrics, compare_models
from .inference import VideoProcessor, process_video, RealtimeDetector
from .utils import  *

__all__ = [
    '__version__',
    '__author__',
    'dataset',
    'models',
    'training',
    'evaluation',
    'utils',
    'inference',
    'get_dataloader',
    'MinecraftMobsDataset',
    'get_model',
    'MODEL_REGISTRY',
    'train_model',
    'evaluate_model',
    'save_metrics',
    'compare_models',
    'VideoProcessor',
    'process_video',
    'RealtimeDetector',
    'plot_training_history',
    'set_seed',
    'get_device'
]
