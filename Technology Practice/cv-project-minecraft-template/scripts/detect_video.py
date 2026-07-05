"""Скрипт для детекции мобов на видео"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import argparse
import yaml
from src.models.factory import create_detector
from src.inference import process_video, RealtimeDetector


def main():
    parser = argparse.ArgumentParser(description="Minecraft Mobs Video Detection")

    parser.add_argument('--input', type=str, required=True,
                        help='Path to input video')
    parser.add_argument('--output', type=str, default='outputs/videos/result.mp4',
                        help='Path to output video')
    parser.add_argument('--model', type=str, default='yolo',
                        choices=['yolo', 'faster_rcnn', 'detr', 'rt_detr', 'yolo_world'],
                        help='Model to use')
    parser.add_argument('--weights', type=str, required=True,
                        help='Path to model weights')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Config file')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Confidence threshold')
    parser.add_argument('--skip-frames', type=int, default=1,
                        help='Process every N-th frame')
    parser.add_argument('--resize', type=str, default=None,
                        help='Resize output (WIDTHxHEIGHT)')
    parser.add_argument('--no-save', action='store_true',
                        help='Do not save output video')

    args = parser.parse_args()

    # Загружаем конфиг
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Создаем модель
    model = create_detector(
        args.model,
        config['models'].get(args.model, {}),
        config['data']['class_names']
    )
    model.build_model()
    model.load(args.weights)

    # Парсим resize
    resize = None
    if args.resize:
        w, h = map(int, args.resize.split('x'))
        resize = (w, h)

    # Обрабатываем видео
    stats = process_video(
        model=model,
        video_path=args.input,
        output_path=args.output,
        class_names=config['data']['class_names'],
        conf_threshold=args.conf,
        skip_frames=args.skip_frames,
        resize=resize,
        save_output=not args.no_save
    )


if __name__ == '__main__':
    main()
