import argparse
import yaml
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any
import json
from datetime import datetime

# Добавляем src в путь
sys.path.append(str(Path(__file__).parent))

from src.dataset.dataset import get_dataloader
from src.models import get_model, MODEL_REGISTRY
from src.training.train import train_model
from src.evaluation.metrics import evaluate_model, save_metrics, compare_models
from src.utils.utils import (
    plot_training_history,
    visualize_predictions,
    setup_logging,
    save_config,
    create_comparison_table
)


def setup_argparse() -> argparse.ArgumentParser:
    """Настройка парсера аргументов командной строки"""
    parser = argparse.ArgumentParser(
        description='Minecraft Mobs Detection - Training & Evaluation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python main.py --model yolo
  python main.py --model all --epochs 50
  python main.py --model yolo,faster_rcnn --config configs/custom.yaml
  python main.py --model all --device cpu --no-train
        """
    )

    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Модели для обучения: yolo, faster_rcnn, detr, rt_detr, yolo_world или all'
    )

    parser.add_argument(
        '--config',
        type=str,
        default='configs/default.yaml',
        help='Путь к конфигурационному файлу (по умолчанию: configs/default.yaml)'
    )

    parser.add_argument(
        '--device',
        type=str,
        default=None,
        choices=['cuda', 'cpu', 'mps'],
        help='Устройство для обучения (переопределяет конфиг)'
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Количество эпох (переопределяет конфиг)'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Размер батча (переопределяет конфиг)'
    )

    parser.add_argument(
        '--lr',
        type=float,
        default=None,
        help='Learning rate (переопределяет конфиг)'
    )

    parser.add_argument(
        '--no-train',
        action='store_true',
        help='Пропустить обучение (только оценка)'
    )

    parser.add_argument(
        '--no-eval',
        action='store_true',
        help='Пропустить оценку (только обучение)'
    )

    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Путь к чекпоинту для возобновления обучения'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Подробный вывод'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed (переопределяет конфиг)'
    )

    parser.add_argument(
        '--num-workers',
        type=int,
        default=None,
        help='Количество workers для DataLoader'
    )

    return parser


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Загрузка конфигурации из YAML файла

    Args:
        config_path: Путь к YAML файлу

    Returns:
        Словарь с конфигурацией
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    logging.info(f"Config loaded from {config_path}")
    return config


def parse_models(models_str: str) -> List[str]:
    """
    Парсинг строки с моделями

    Args:
        models_str: Строка с перечислением моделей или 'all'

    Returns:
        Список названий моделей
    """
    if models_str.lower() == 'all':
        return list(MODEL_REGISTRY.keys())

    models = [m.strip().lower() for m in models_str.split(',')]

    # Проверяем, что все модели существуют
    for model in models:
        if model not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model: {model}. Available: {list(MODEL_REGISTRY.keys())}")

    return models


def override_config(config: Dict, args: argparse.Namespace) -> Dict:
    """
    Переопределение параметров конфига из аргументов командной строки

    Args:
        config: Исходный конфиг
        args: Аргументы командной строки

    Returns:
        Обновленный конфиг
    """
    # Переопределение общих параметров
    if args.device:
        config['training']['device'] = args.device
        logging.info(f"Device overridden to: {args.device}")

    if args.seed:
        config['training']['seed'] = args.seed

    if args.num_workers:
        config['training']['num_workers'] = args.num_workers

    if args.batch_size:
        config['data']['batch_size'] = args.batch_size

    # Переопределение параметров для конкретных моделей
    models = parse_models(args.model)
    for model_name in models:
        if model_name in config['models']:
            if args.epochs:
                config['models'][model_name]['epochs'] = args.epochs
            if args.lr:
                config['models'][model_name]['lr'] = args.lr

    return config


def train_and_evaluate_model(
        model_name: str,
        model_config: Dict,
        data_config: Dict,
        training_config: Dict,
        evaluation_config: Dict,
        results_config: Dict,
        args: argparse.Namespace,
        experiment_dir: Path
) -> Dict:
    """
    Обучение и оценка одной модели

    Args:
        model_name: Название модели
        model_config: Конфигурация модели
        data_config: Конфигурация данных
        training_config: Конфигурация обучения
        evaluation_config: Конфигурация оценки
        results_config: Конфигурация результатов
        args: Аргументы командной строки
        experiment_dir: Директория эксперимента

    Returns:
        Словарь с метриками
    """
    logging.info(f"{'=' * 60}")
    logging.info(f"Processing model: {model_name.upper()}")
    logging.info(f"{'=' * 60}")

    # 1. Создание даталоадеров
    logging.info("Creating data loaders...")
    train_loader = get_dataloader(
        data_config,
        split='train',
        batch_size=data_config.get('batch_size', 16),
        num_workers=training_config.get('num_workers', 4)
    )
    val_loader = get_dataloader(
        data_config,
        split='val',
        batch_size=data_config.get('batch_size', 16),
        num_workers=training_config.get('num_workers', 4),
        shuffle=False
    )

    logging.info(f"Train samples: {len(train_loader.dataset)}")
    logging.info(f"Val samples: {len(val_loader.dataset)}")

    # 2. Создание модели
    logging.info(f"Creating {model_name} model...")
    model = get_model(
        model_name=model_name,
        config=model_config,
        class_names=data_config['class_names']
    )
    model.build_model()

    # Вывод информации о модели
    model_info = model.get_model_info()
    logging.info(f"Model info: {json.dumps(model_info, indent=2)}")

    # 3. Обучение (если не отключено)
    history = None
    if not args.no_train:
        logging.info("Starting training...")

        # Объединяем все конфиги для обучения
        train_config = {
            **model_config,
            **training_config,
            'seed': training_config.get('seed', 42),
            'device': training_config.get('device', 'cuda')
        }

        if args.resume:
            logging.info(f"Resuming from checkpoint: {args.resume}")
            model.load(args.resume)

        try:
            history = train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                config=train_config,
                model_name=model_name,
                save_dir=experiment_dir / 'checkpoints'
            )

            logging.info(f"Training completed. Best mAP@0.5: {history.get('mAP_0.5', 'N/A')}")

        except Exception as e:
            logging.error(f"Training failed for {model_name}: {e}", exc_info=True)
            if not args.no_eval:
                logging.warning("Skipping evaluation due to training failure")
                return None
    else:
        logging.info("Training skipped (--no-train)")
        # Загружаем предобученную модель
        checkpoint_path = args.resume or experiment_dir / 'checkpoints' / f'{model_name}_best.pt'
        if Path(checkpoint_path).exists():
            model.load(str(checkpoint_path))
            logging.info(f"Loaded model from {checkpoint_path}")

    # 4. Оценка (если не отключена)
    metrics = None
    if not args.no_eval:
        logging.info("Evaluating model...")

        try:
            metrics = evaluate_model(
                model=model,
                data_loader=val_loader,
                config=evaluation_config,
                class_names=data_config['class_names']
            )

            # Сохранение метрик
            save_metrics(
                metrics=metrics,
                model_name=model_name,
                results_config=results_config,
                experiment_dir=experiment_dir
            )

            logging.info(f"Evaluation metrics: {json.dumps(metrics, indent=2)}")

        except Exception as e:
            logging.error(f"Evaluation failed for {model_name}: {e}", exc_info=True)

    # 5. Визуализация
    if history and not args.no_train:
        logging.info("Generating training plots...")
        try:
            plot_training_history(
                history=history,
                model_name=model_name,
                results_config=results_config,
                save_dir=experiment_dir / 'plots'
            )
        except Exception as e:
            logging.error(f"Plotting failed: {e}")

    if metrics and not args.no_eval:
        logging.info("Generating prediction visualizations...")
        try:
            visualize_predictions(
                model=model,
                data_loader=val_loader,
                model_name=model_name,
                results_config=results_config,
                num_images=5,
                save_dir=experiment_dir / 'predictions',
                class_names=data_config['class_names']
            )
        except Exception as e:
            logging.error(f"Visualization failed: {e}")

    # 6. Сохранение модели
    if not args.no_train:
        model_path = experiment_dir / 'models' / f'{model_name}_final.pt'
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(model_path))
        logging.info(f"Model saved to {model_path}")

    return metrics


def main():
    """Главная функция"""

    # Парсинг аргументов
    parser = setup_argparse()
    args = parser.parse_args()

    # Создание директории эксперимента
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = Path('results') / f'experiment_{timestamp}'
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # Настройка логирования
    setup_logging(
        log_dir=experiment_dir / 'logs',
        verbose=args.verbose
    )

    logging.info("=" * 70)
    logging.info("Minecraft Mobs Detection - Starting Experiment")
    logging.info(f"Timestamp: {timestamp}")
    logging.info(f"Command: {' '.join(sys.argv)}")
    logging.info("=" * 70)

    try:
        # Загрузка конфига
        config = load_config(args.config)

        # Переопределение параметров из аргументов
        config = override_config(config, args)

        # Сохранение финального конфига
        save_config(config, experiment_dir / 'config.yaml')
        logging.info(f"Config saved to {experiment_dir / 'config.yaml'}")

        # Парсинг моделей
        models_to_train = parse_models(args.model)
        logging.info(f"Models to process: {', '.join(models_to_train)}")

        # Словарь для хранения результатов
        all_metrics = {}

        # Обучение и оценка каждой модели
        for model_name in models_to_train:
            if model_name not in config['models']:
                logging.warning(f"Model {model_name} not found in config, skipping...")
                continue

            model_config = config['models'][model_name]
            data_config = config['data']
            training_config = config['training']
            evaluation_config = config['evaluation']
            results_config = config['results']

            metrics = train_and_evaluate_model(
                model_name=model_name,
                model_config=model_config,
                data_config=data_config,
                training_config=training_config,
                evaluation_config=evaluation_config,
                results_config=results_config,
                args=args,
                experiment_dir=experiment_dir
            )

            if metrics:
                all_metrics[model_name] = metrics

        # 7. Сравнительный анализ
        if len(all_metrics) > 1:
            logging.info("\n" + "=" * 70)
            logging.info("COMPARATIVE ANALYSIS")
            logging.info("=" * 70)

            # Создание таблицы сравнения
            comparison_table = create_comparison_table(all_metrics)
            logging.info(f"\n{comparison_table}")

            # Сохранение результатов сравнения
            compare_models(
                all_metrics=all_metrics,
                results_config=results_config,
                save_dir=experiment_dir / 'comparison'
            )

            # Вывод лучшей модели
            best_model = max(all_metrics.items(), key=lambda x: x[1].get('mAP_50', 0))
            logging.info(f"\nBest model (mAP@0.5): {best_model[0]} with {best_model[1]['mAP_50']:.4f}")

            best_model_iou = max(all_metrics.items(), key=lambda x: x[1].get('mAP_50_95', 0))
            logging.info(f"Best model (mAP@0.5:0.95): {best_model_iou[0]} with {best_model_iou[1]['mAP_50_95']:.4f}")

        elif len(all_metrics) == 1:
            model_name = list(all_metrics.keys())[0]
            logging.info(f"\nSingle model results for {model_name}:")
            logging.info(json.dumps(all_metrics[model_name], indent=2))

        else:
            logging.warning("No metrics available for comparison")

        logging.info("\n" + "=" * 70)
        logging.info(f"Experiment completed successfully!")
        logging.info(f"Results saved to: {experiment_dir}")
        logging.info("=" * 70)

    except Exception as e:
        logging.error(f"Experiment failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
