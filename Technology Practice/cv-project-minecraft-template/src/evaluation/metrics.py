"""
Метрики для оценки моделей детекции объектов.
Поддерживает вычисление mAP, Precision, Recall, F1-Score для object detection.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from collections import defaultdict
import json
from pathlib import Path
import logging
import time
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
from torchmetrics.detection import MeanAveragePrecision
import torch

logger = logging.getLogger(__name__)


@dataclass
class DetectionMetrics:
    """Класс для хранения метрик детекции"""
    mAP_50: float = 0.0
    mAP_75: float = 0.0
    mAP_50_95: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    inference_time: float = 0.0  # среднее время инференса в мс
    fps: float = 0.0
    num_parameters: int = 0
    per_class_metrics: Dict[str, Dict[str, float]] = None

    def to_dict(self) -> Dict:
        return asdict(self)


def calculate_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Вычисление IoU (Intersection over Union) между двумя боксами

    Args:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]

    Returns:
        IoU score
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # Площадь пересечения
    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    # Площади боксов
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Площадь объединения
    union = box1_area + box2_area - intersection

    return intersection / union if union > 0 else 0.0


def calculate_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """
    Вычисление Average Precision (AP) методом интерполяции

    Args:
        recalls: Массив recall значений
        precisions: Массив precision значений

    Returns:
        AP score
    """
    # Добавляем граничные точки
    recalls = np.concatenate(([0.0], recalls, [1.0]))
    precisions = np.concatenate(([0.0], precisions, [0.0]))

    # Интерполяция precision (берем максимум справа)
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])

    # Вычисляем AP как площадь под кривой
    indices = np.where(recalls[1:] != recalls[:-1])[0]
    ap = np.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1])

    return ap


def calculate_map(
        predictions: List[Dict],
        targets: List[Dict],
        iou_thresholds: List[float] = None,
        num_classes: int = 6,
        class_names: List[str] = None
) -> Dict:
    """
    Вычисление mAP (mean Average Precision) для детекции объектов

    Args:
        predictions: Список предсказаний в формате:
            [{'boxes': [[x1,y1,x2,y2], ...],
              'scores': [0.9, ...],
              'labels': [0, ...]}, ...]
        targets: Список целевых аннотаций в том же формате
        iou_thresholds: Пороги IoU для mAP (по умолчанию [0.5, 0.75, 0.5:0.95])
        num_classes: Количество классов
        class_names: Имена классов

    Returns:
        Словарь с метриками
    """
    if iou_thresholds is None:
        iou_thresholds = [0.5, 0.75] + list(np.arange(0.5, 1.0, 0.05))

    # Собираем все детекции по классам
    class_predictions = defaultdict(list)
    class_targets = defaultdict(list)

    for pred, target in zip(predictions, targets):
        # Предсказания
        for box, score, label in zip(
                pred.get('boxes', []),
                pred.get('scores', []),
                pred.get('labels', [])
        ):
            class_predictions[label].append({
                'image_id': len(class_predictions[label]),
                'box': box,
                'score': score
            })

        # Ground truth
        for box, label in zip(
                target.get('boxes', []),
                target.get('labels', [])
        ):
            if isinstance(label, torch.Tensor):
                label = label.item()
            class_targets[label].append({
                'image_id': len(class_targets[label]),
                'box': box,
                'matched': False
            })

    # Вычисляем AP для каждого класса и каждого порога IoU
    ap_per_class = defaultdict(lambda: defaultdict(float))

    for class_id in range(num_classes):
        preds = class_predictions[class_id]
        gts = class_targets[class_id]

        if len(preds) == 0 and len(gts) == 0:
            continue

        # Сортируем предсказания по уверенности
        preds = sorted(preds, key=lambda x: x['score'], reverse=True)

        for iou_thresh in iou_thresholds:
            # Сбрасываем метки matched
            for gt in gts:
                gt['matched'] = False

            tp = np.zeros(len(preds))
            fp = np.zeros(len(preds))

            for i, pred in enumerate(preds):
                best_iou = 0
                best_gt_idx = -1

                # Ищем лучший matching ground truth
                for j, gt in enumerate(gts):
                    if gt['image_id'] == pred['image_id'] and not gt['matched']:
                        iou = calculate_iou(pred['box'], gt['box'])
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = j

                if best_iou >= iou_thresh:
                    tp[i] = 1
                    gts[best_gt_idx]['matched'] = True
                else:
                    fp[i] = 1

            # Кумулятивные суммы
            tp_cumsum = np.cumsum(tp)
            fp_cumsum = np.cumsum(fp)

            # Precision и Recall
            recalls = tp_cumsum / max(len(gts), 1)
            precisions = tp_cumsum / np.maximum(tp_cumsum + fp_cumsum, np.finfo(np.float64).eps)

            # AP
            ap = calculate_ap(recalls, precisions)
            ap_per_class[class_id][f'AP@{iou_thresh:.2f}'] = ap

    # Усредняем по классам
    metrics = {}

    # mAP@0.5
    ap_50 = [ap_per_class[c].get('AP@0.50', 0) for c in range(num_classes)
             if len(class_targets[c]) > 0 or len(class_predictions[c]) > 0]
    metrics['mAP_50'] = np.mean(ap_50) if ap_50 else 0.0

    # mAP@0.75
    ap_75 = [ap_per_class[c].get('AP@0.75', 0) for c in range(num_classes)
             if len(class_targets[c]) > 0 or len(class_predictions[c]) > 0]
    metrics['mAP_75'] = np.mean(ap_75) if ap_75 else 0.0

    # mAP@0.5:0.95
    ap_range = []
    for c in range(num_classes):
        if len(class_targets[c]) > 0 or len(class_predictions[c]) > 0:
            aps = [ap_per_class[c].get(f'AP@{t:.2f}', 0) for t in np.arange(0.5, 1.0, 0.05)]
            ap_range.append(np.mean(aps))
    metrics['mAP_50_95'] = np.mean(ap_range) if ap_range else 0.0

    # Per-class метрики
    if class_names:
        per_class = {}
        for c in range(num_classes):
            if c < len(class_names):
                per_class[class_names[c]] = {
                    'AP@0.50': ap_per_class[c].get('AP@0.50', 0),
                    'AP@0.75': ap_per_class[c].get('AP@0.75', 0),
                    'AP@0.50:0.95': np.mean([
                        ap_per_class[c].get(f'AP@{t:.2f}', 0)
                        for t in np.arange(0.5, 1.0, 0.05)
                    ]) if c in ap_per_class else 0
                }
        metrics['per_class'] = per_class

    return metrics


def calculate_precision_recall(
        predictions: List[Dict],
        targets: List[Dict],
        iou_threshold: float = 0.5,
        conf_threshold: float = 0.25
) -> Dict[str, float]:
    """
    Вычисление Precision, Recall и F1-Score для детекции

    Args:
        predictions: Список предсказаний
        targets: Список аннотаций
        iou_threshold: Порог IoU
        conf_threshold: Порог уверенности

    Returns:
        Словарь с precision, recall, f1
    """
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for pred, target in zip(predictions, targets):
        pred_boxes = [b for b, s in zip(pred.get('boxes', []), pred.get('scores', []))
                      if s >= conf_threshold]
        gt_boxes = target.get('boxes', [])

        # Сопоставление предсказаний с ground truth
        matched_gt = set()
        tp = 0
        fp = 0

        for pred_box in pred_boxes:
            best_iou = 0
            best_gt_idx = -1

            for j, gt_box in enumerate(gt_boxes):
                if j not in matched_gt:
                    iou = calculate_iou(pred_box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = j

            if best_iou >= iou_threshold:
                tp += 1
                matched_gt.add(best_gt_idx)
            else:
                fp += 1

        fn = len(gt_boxes) - len(matched_gt)

        total_tp += tp
        total_fp += fp
        total_fn += fn

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }


def evaluate_model(
        model,
        data_loader: torch.utils.data.DataLoader,
        config: Dict,
        class_names: List[str] = None,
        device: str = None
) -> DetectionMetrics:
    """
    Полная оценка модели детекции

    Args:
        model: Модель детекции (должна иметь метод predict)
        data_loader: DataLoader с валидационными данными
        config: Конфигурация оценки
        class_names: Имена классов
        device: Устройство для вычислений

    Returns:
        DetectionMetrics объект с метриками
    """
    logger.info("Starting model evaluation...")

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if hasattr(model, 'model'):
        model.model.eval()
    else:
        model.eval()


    all_predictions = []
    all_targets = []
    inference_times = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            # Получаем изображения и аннотации
            if isinstance(batch, dict):
                images = batch['image']
                targets = batch.get('targets', batch.get('labels', None))
            elif isinstance(batch, (list, tuple)):
                images = batch[0]
                targets = batch[1] if len(batch) > 1 else None
            else:
                images = batch
                targets = None

            # Обрабатываем каждое изображение в батче
            for i in range(len(images)):
                image = images[i]

                # Конвертация в numpy если нужно
                if isinstance(image, torch.Tensor):
                    if image.shape[0] == 3:  # CxHxW
                        # Денормализация
                        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                        image_np = (image * std + mean).clamp(0, 1)
                        image_np = image_np.permute(1, 2, 0).cpu().numpy()
                    else:
                        image_np = image.cpu().numpy()
                else:
                    image_np = image

                # Инференс с замером времени
                start_time = time.time()
                prediction = model.predict(
                    image_np,
                    conf_threshold=config.get('conf_threshold', 0.25)
                )
                inference_time = (time.time() - start_time) * 1000  # в мс
                inference_times.append(inference_time)

                all_predictions.append(prediction)

                # Ground truth
                if targets is not None:
                    if isinstance(targets, dict):
                        if 'boxes' in targets:
                            # Берем i-й элемент из батча
                            target = {
                                'boxes': targets['boxes'][i].cpu().numpy() if isinstance(targets['boxes'],
                                                                                         torch.Tensor) else
                                targets['boxes'][i],
                                'labels': targets['labels'][i].cpu().numpy() if isinstance(targets['labels'],
                                                                                           torch.Tensor) else
                                targets['labels'][i]
                            }
                        else:
                            target = {'boxes': [], 'labels': []}
                    elif isinstance(targets, (list, torch.Tensor)):
                        target = {
                            'boxes': targets[i].get('boxes', []) if isinstance(targets[i], dict) else [],
                            'labels': targets[i].get('labels', []) if isinstance(targets[i], dict) else []
                        }
                    else:
                        target = {'boxes': [], 'labels': []}

                    all_targets.append(target)
                else:
                    all_targets.append({'boxes': [], 'labels': []})

            # Прогресс
            if batch_idx % 10 == 0:
                logger.debug(f"Evaluated {batch_idx * len(images)} samples...")

    logger.info(f"Processed {len(all_predictions)} images")

    # Вычисление метрик
    metrics = DetectionMetrics()

    # mAP
    iou_threshold = config.get('iou_threshold', 0.5)
    iou_thresholds = [iou_threshold] + list(np.arange(0.5, 1.0, 0.05))

    num_classes = len(class_names) if class_names else config.get('num_classes', 6)

    map_metrics = calculate_map(
        predictions=all_predictions,
        targets=all_targets,
        iou_thresholds=iou_thresholds,
        num_classes=num_classes,
        class_names=class_names
    )

    metrics.mAP_50 = map_metrics.get('mAP_50', 0.0)
    metrics.mAP_75 = map_metrics.get('mAP_75', 0.0)
    metrics.mAP_50_95 = map_metrics.get('mAP_50_95', 0.0)

    # Precision, Recall, F1
    pr_metrics = calculate_precision_recall(
        predictions=all_predictions,
        targets=all_targets,
        iou_threshold=config.get('iou_threshold', 0.5),
        conf_threshold=config.get('conf_threshold', 0.25)
    )

    metrics.precision = pr_metrics['precision']
    metrics.recall = pr_metrics['recall']
    metrics.f1_score = pr_metrics['f1_score']

    # Время инференса
    metrics.inference_time = np.mean(inference_times) if inference_times else 0.0
    metrics.fps = 1000 / metrics.inference_time if metrics.inference_time > 0 else 0.0

    # Параметры модели
    try:
        metrics.num_parameters = model.count_parameters()
    except:
        metrics.num_parameters = 0

    # Per-class метрики
    if 'per_class' in map_metrics:
        metrics.per_class_metrics = map_metrics['per_class']

    logger.info(f"Evaluation complete. mAP@0.5: {metrics.mAP_50:.4f}, "
                f"mAP@0.5:0.95: {metrics.mAP_50_95:.4f}, "
                f"FPS: {metrics.fps:.1f}")

    return metrics


def save_metrics(
        metrics: Union[DetectionMetrics, Dict],
        model_name: str,
        config: Dict,
        experiment_dir: Optional[Path] = None
) -> None:
    """
    Сохранение метрик в JSON и текстовый файл

    Args:
        metrics: Метрики (объект DetectionMetrics или словарь)
        model_name: Название модели
        config: Конфигурация результатов
        experiment_dir: Директория эксперимента
    """
    # Определяем директорию для сохранения
    if experiment_dir:
        save_dir = Path(experiment_dir) / 'metrics'
    elif isinstance(config, dict):
        save_dir = Path(config.get('results', {}).get('logs_dir', 'results/logs'))
    else:
        save_dir = Path('results/logs')

    save_dir.mkdir(parents=True, exist_ok=True)

    # Конвертируем в словарь
    if isinstance(metrics, DetectionMetrics):
        metrics_dict = metrics.to_dict()
    else:
        metrics_dict = metrics

    # Сохраняем JSON
    json_path = save_dir / f"{model_name}_metrics.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(metrics_dict, f, indent=2, ensure_ascii=False)

    # Сохраняем человекочитаемый отчет
    report_path = save_dir / f"{model_name}_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"Model Evaluation Report - {model_name.upper()}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"mAP@0.5:      {metrics_dict.get('mAP_50', 0):.4f}\n")
        f.write(f"mAP@0.75:     {metrics_dict.get('mAP_75', 0):.4f}\n")
        f.write(f"mAP@0.5:0.95: {metrics_dict.get('mAP_50_95', 0):.4f}\n")
        f.write(f"Precision:    {metrics_dict.get('precision', 0):.4f}\n")
        f.write(f"Recall:       {metrics_dict.get('recall', 0):.4f}\n")
        f.write(f"F1-Score:     {metrics_dict.get('f1_score', 0):.4f}\n")
        f.write(f"FPS:          {metrics_dict.get('fps', 0):.1f}\n")
        f.write(f"Inference:    {metrics_dict.get('inference_time', 0):.1f} ms\n")
        f.write(f"Parameters:   {metrics_dict.get('num_parameters', 0):,}\n")

        # Per-class метрики
        if metrics_dict.get('per_class_metrics'):
            f.write("\nPer-Class Metrics:\n")
            f.write("-" * 60 + "\n")
            for class_name, class_metrics in metrics_dict['per_class_metrics'].items():
                f.write(f"\n{class_name}:\n")
                for metric_name, value in class_metrics.items():
                    f.write(f"  {metric_name}: {value:.4f}\n")

    logger.info(f"Metrics saved to {save_dir}")


def compare_models(
        all_metrics: Dict[str, Union[DetectionMetrics, Dict]],
        results_config: Dict,
        save_dir: Optional[Path] = None
) -> Dict:
    """
    Сравнение метрик нескольких моделей

    Args:
        all_metrics: Словарь {имя_модели: метрики}
        results_config: Конфигурация результатов
        save_dir: Директория для сохранения

    Returns:
        Словарь с результатами сравнения
    """
    if save_dir:
        comparison_dir = Path(save_dir)
    elif isinstance(results_config, dict):
        comparison_dir = Path(results_config.get('results', {}).get('plots_dir', 'results/plots'))
    else:
        comparison_dir = Path('results/comparison')

    comparison_dir.mkdir(parents=True, exist_ok=True)

    # Собираем метрики
    comparison = {}
    for model_name, metrics in all_metrics.items():
        if isinstance(metrics, DetectionMetrics):
            comparison[model_name] = metrics.to_dict()
        else:
            comparison[model_name] = metrics

    # Сохраняем сравнение
    with open(comparison_dir / 'model_comparison.json', 'w', encoding='utf-8') as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)

    # Находим лучшую модель по разным метрикам
    best_models = {}
    metrics_to_compare = ['mAP_50', 'mAP_50_95', 'f1_score', 'fps']

    for metric in metrics_to_compare:
        values = {name: m.get(metric, 0) for name, m in comparison.items()}
        if values:
            best_model = max(values.items(), key=lambda x: x[1])
            best_models[metric] = {
                'model': best_model[0],
                'value': best_model[1]
            }

    # Сохраняем лучшие модели
    with open(comparison_dir / 'best_models.json', 'w', encoding='utf-8') as f:
        json.dump(best_models, f, indent=2, ensure_ascii=False)

    # Выводим результаты
    logger.info("\n" + "=" * 60)
    logger.info("BEST MODELS")
    logger.info("=" * 60)
    for metric, info in best_models.items():
        logger.info(f"Best {metric}: {info['model']} = {info['value']:.4f}")

    return comparison


def plot_metrics_comparison(
        all_metrics: Dict[str, Dict],
        save_dir: Path,
        figsize: Tuple[int, int] = (12, 8)
) -> None:
    """
    Визуализация сравнения метрик моделей

    Args:
        all_metrics: Словарь с метриками
        save_dir: Директория для сохранения
        figsize: Размер фигуры
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    metrics_names = ['mAP_50', 'mAP_50_95', 'precision', 'recall', 'f1_score']
    model_names = list(all_metrics.keys())

    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(metrics_names))
    width = 0.8 / len(model_names)

    colors = plt.cm.Set2(np.linspace(0, 1, len(model_names)))

    for i, model_name in enumerate(model_names):
        values = [all_metrics[model_name].get(m, 0) for m in metrics_names]
        bars = ax.bar(x + i * width, values, width,
                      label=model_name.upper(),
                      color=colors[i],
                      edgecolor='white',
                      linewidth=1.5)

        # Добавляем значения над барами
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f'{val:.3f}',
                    ha='center', va='bottom',
                    fontsize=8, fontweight='bold')

    ax.set_xlabel('Metrics', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Comparison - Minecraft Mobs Detection', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * (len(model_names) - 1) / 2)
    ax.set_xticklabels(metrics_names, fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1.05])

    plt.tight_layout()
    plt.savefig(save_dir / 'metrics_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Metrics comparison plot saved to {save_dir}")

def compute_detection_metrics(all_preds, all_targets):
    """
    Вычисляет метрики детекции: mAP@0.5, mAP@0.5:0.95, Precision, Recall, F1.

    Args:
        all_preds: Список предсказаний (boxes, scores, labels)
        all_targets: Список истинных данных (boxes, labels)

    Returns:
        dict: Словарь с метриками
    """
    if not all_preds or not all_targets:
        return {
            'mAP_0.5': 0.0,
            'mAP_0.5_0.95': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0
        }

    metric = MeanAveragePrecision(iou_type='bbox')
    metric.update(all_preds, all_targets)
    map_metrics = metric.compute()

    precision = map_metrics.get('precision', torch.tensor(0.0)).mean().item()
    recall = map_metrics.get('recall', torch.tensor(0.0)).mean().item()
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

    return {
        'mAP_0.5': map_metrics['map_50'].item(),
        'mAP_0.5_0.95': map_metrics['map'].item(),
        'precision': precision,
        'recall': recall,
        'f1': f1
    }