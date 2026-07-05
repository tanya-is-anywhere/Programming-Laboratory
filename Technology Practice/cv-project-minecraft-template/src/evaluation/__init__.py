"""
Модуль для оценки качества моделей детекции
"""

from .metrics import (
    evaluate_model,
    save_metrics,
    compare_models,
    DetectionMetrics,
    calculate_map,
    calculate_precision_recall,
    calculate_iou,
    calculate_ap,
    plot_metrics_comparison
)

__all__ = [
    'evaluate_model',
    'save_metrics',
    'compare_models',
    'DetectionMetrics',
    'calculate_map',
    'calculate_precision_recall',
    'calculate_iou',
    'calculate_ap',
    'plot_metrics_comparison'
]
