"""
main.py - Главная точка входа для обучения моделей

Примеры запуска:
    python main.py --model yolo --epochs 50
    python main.py --model yolo,rtdetr --epochs 30 --batch_size 8
    python main.py --all --epochs 100
"""

import argparse
import yaml
from pathlib import Path

from src.training.trainer import train_model, train_models
from src.utils.utils import *

def parse_args():
    parser = argparse.ArgumentParser(
        description='Обучение моделей детекции мобов Minecraft'
    )

    parser.add_argument(
        '--model', '-m',
        type=str,
        default=None,
        help='Имя модели или список через запятую (yolo, yolo_world, faster_rcnn, detr, rtdetr)'
    )

    parser.add_argument(
        '--all',
        action='store_true',
        help='Обучить все модели'
    )

    parser.add_argument(
        '--epochs', '-e',
        type=int,
        default=None,
        help='Количество эпох (переопределяет конфиг)'
    )

    parser.add_argument(
        '--batch_size', '-b',
        type=int,
        default=None,
        help='Размер батча (переопределяет конфиг)'
    )

    parser.add_argument(
        '--image_size', '-i',
        type=int,
        default=None,
        help='Размер изображения (переопределяет конфиг)'
    )

    parser.add_argument(
        '--device', '-d',
        type=str,
        default=None,
        choices=['cpu', 'cuda'],
        help='Устройство для обучения (переопределяет конфиг)'
    )

    parser.add_argument(
        '--config', '-c',
        type=str,
        default='configs/default.yaml',
        help='Путь к конфигурационному файлу'
    )

    return parser.parse_args()


def get_models_list(args):
    """Возвращает список моделей для обучения."""
    if args.all:
        return ['yolo', 'yolo_world', 'faster_rcnn', 'detr', 'rt_detr']
    if args.model:
        return [m.strip() for m in args.model.split(',')]
    raise ValueError(
        "Укажите модель через --model или --all\n"
        "Доступные модели: yolo, yolo_world, faster_rcnn, detr, rtdetr"
    )


def main():
    args = parse_args()
    model_names = get_models_list(args)
    log_file = setup_file_logging(log_dir='results/logs')

    logger.info("=" * 60)
    logger.info("ЗАПУСК ОБУЧЕНИЯ")
    logger.info("=" * 60)
    logger.info(f"Модели: {', '.join(model_names)}")

    if args.epochs:
        logger.info(f"Эпохи: {args.epochs}")
    if args.batch_size:
        logger.info(f"Батч: {args.batch_size}")
    if args.image_size:
        logger.info(f"Размер изображения: {args.image_size}")
    if args.device:
        logger.info(f"Устройство: {args.device}")

    config = load_config(args.config)

    override_params = {}
    if args.epochs is not None:
        override_params['epochs'] = args.epochs
    if args.batch_size is not None:
        override_params['batch_size'] = args.batch_size
    if args.image_size is not None:
        override_params['image_size'] = args.image_size
    if args.device is not None:
        override_params['device'] = args.device

    # Запускаем обучение
    if len(model_names) == 1:
        results = train_model(
            model_names[0],
            config,
            override_params=override_params
        )
    else:
        results = train_models(
            model_names,
            config_path=args.config,
            override_params=override_params
        )

    logger.info("=" * 60)
    logger.info("ВСЕ МОДЕЛИ ОБУЧЕНЫ!")
    logger.info("=" * 60)
    comparison_metrics = {}
    for model_name, res in results.items():
        if isinstance(res, dict):
            model_path = res.get('model_path', 'неизвестно')
            history = res.get('history', {})
            # Берём последние значения метрик
            def get_metric(history, *keys):
                """Ищет метрику по нескольким возможным ключам."""
                for key in keys:
                    val = history.get(key, [0])
                    if isinstance(val, list) and val:
                        return max(val) if any(v != 0 for v in val) else 0
                    elif val:
                        return val
                return 0

            comparison_metrics[model_name] = {
                'mAP_0.5': get_metric(history, 'metrics/mAP50(B)', 'mAP_0.5'),
                'mAP_0.5_0.95': get_metric(history, 'metrics/mAP50-95(B)', 'mAP_0.5_0.95'),
                'precision': get_metric(history, 'metrics/precision(B)', 'precision'),
                'recall': get_metric(history, 'metrics/recall(B)', 'recall'),
                'f1_score': get_metric(history, 'metrics/f1_score', 'f1')
            }


        else:
            model_path = str(res)
            comparison_metrics[model_name] = {}

        logger.info(f"  {model_name}: {model_path}")
    if comparison_metrics:
        table = create_comparison_table(comparison_metrics)
        logger.info(table)
    logger.info(f"Лог-файл: {log_file}")

if __name__ == '__main__':
    main()