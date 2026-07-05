"""
Обработка видео с детекцией мобов Minecraft
"""

import cv2
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Generator
import logging
import time
from dataclasses import dataclass
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class FrameResult:
    """Результат обработки одного кадра"""
    frame: np.ndarray
    frame_number: int
    timestamp: float
    detections: List[Dict]
    fps: float
    processing_time: float


class VideoProcessor:
    """
    Класс для обработки видео с детекцией объектов

    Поддерживает:
    - Чтение видеофайлов
    - Детекцию на каждом N-м кадре
    - Трекинг объектов между кадрами
    - Сохранение результатов
    - Подсчет статистики
    """

    def __init__(
            self,
            model,
            class_names: List[str],
            conf_threshold: float = 0.25,
            iou_threshold: float = 0.45,
            device: str = 'cuda',
            img_size: int = 640
    ):
        """
        Args:
            model: Модель детекции
            class_names: Имена классов
            conf_threshold: Порог уверенности
            iou_threshold: Порог IoU для NMS
            device: Устройство ('cuda' или 'cpu')
            img_size: Размер изображения для модели
        """
        self.model = model
        self.class_names = class_names
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.img_size = img_size

        # Статистика
        self.stats = {
            'frames_processed': 0,
            'total_detections': 0,
            'detections_per_class': {name: 0 for name in class_names},
            'processing_times': [],
            'fps_history': []
        }

        # Кэш для трекинга
        self.track_history = {}
        self.track_counter = 0

        logger.info(f"VideoProcessor initialized with {len(class_names)} classes")
        logger.info(f"Device: {device}, Conf threshold: {conf_threshold}")

    def process_frame(self, frame: np.ndarray, frame_number: int) -> FrameResult:
        """
        Обработка одного кадра

        Args:
            frame: BGR изображение (H, W, 3)
            frame_number: Номер кадра

        Returns:
            FrameResult с детекциями и метаданными
        """
        start_time = time.time()

        # Конвертация BGR -> RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Детекция
        predictions = self.model.predict(
            frame_rgb
        )

        processing_time = (time.time() - start_time) * 1000  # мс
        fps = 1000 / processing_time if processing_time > 0 else 0

        # Отрисовка детекций
        frame_with_boxes = self._draw_boxes(frame, predictions)

        # Обновление статистики
        self._update_stats(predictions, processing_time, fps)

        return FrameResult(
            frame=frame_with_boxes,
            frame_number=frame_number,
            timestamp=frame_number / 30.0,  # предполагаем 30 FPS
            detections=predictions['boxes'],
            fps=fps,
            processing_time=processing_time
        )

    def _draw_boxes(self, frame: np.ndarray, predictions: Dict) -> np.ndarray:
        """
        Отрисовка bounding boxes и меток

        Args:
            frame: BGR изображение
            predictions: Предсказания модели

        Returns:
            Изображение с нарисованными боксами
        """
        frame_copy = frame.copy()

        # Цвета для классов
        colors = self._get_class_colors()

        for box, score, class_name in zip(
                predictions.get('boxes', []),
                predictions.get('scores', []),
                predictions.get('class_names', [])
        ):
            if score < self.conf_threshold:
                continue

            x1, y1, x2, y2 = map(int, box[:4])
            color = colors.get(class_name, (0, 255, 0))

            # Рисуем прямоугольник
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 2)

            # Фон для текста
            label = f"{class_name}: {score:.2f}"
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)

            cv2.rectangle(
                frame_copy,
                (x1, y1 - text_h - 10),
                (x1 + text_w + 10, y1),
                color,
                -1
            )

            # Текст
            cv2.putText(
                frame_copy,
                label,
                (x1 + 5, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2
            )

        # Добавляем FPS и статистику
        info_text = f"Detections: {len(predictions.get('boxes', []))} | "
        info_text += f"Conf: {self.conf_threshold}"

        cv2.putText(
            frame_copy,
            info_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        return frame_copy

    def _get_class_colors(self) -> Dict[str, Tuple[int, int, int]]:
        """Генерация цветов для классов"""
        colors = {}
        base_colors = [
            (0, 255, 0),  # Зеленый
            (255, 0, 0),  # Синий
            (0, 0, 255),  # Красный
            (255, 255, 0),  # Голубой
            (255, 0, 255),  # Пурпурный
            (0, 255, 255),  # Желтый
        ]

        for i, class_name in enumerate(self.class_names):
            colors[class_name] = base_colors[i % len(base_colors)]

        return colors

    def _update_stats(self, predictions: Dict, processing_time: float, fps: float):
        """Обновление статистики"""
        self.stats['frames_processed'] += 1
        self.stats['total_detections'] += len(predictions.get('boxes', []))
        self.stats['processing_times'].append(processing_time)
        self.stats['fps_history'].append(fps)

        # Статистика по классам
        for class_name in predictions.get('class_names', []):
            if class_name in self.stats['detections_per_class']:
                self.stats['detections_per_class'][class_name] += 1

    def get_stats(self) -> Dict:
        """Получение статистики обработки"""
        times = self.stats['processing_times']

        return {
            **self.stats,
            'avg_processing_time': np.mean(times) if times else 0,
            'avg_fps': np.mean(self.stats['fps_history']) if self.stats['fps_history'] else 0,
            'min_fps': np.min(self.stats['fps_history']) if self.stats['fps_history'] else 0,
            'max_fps': np.max(self.stats['fps_history']) if self.stats['fps_history'] else 0,
            'detections_per_frame': self.stats['total_detections'] / max(1, self.stats['frames_processed'])
        }

    def print_stats(self):
        """Вывод статистики"""
        stats = self.get_stats()
        print("\n" + "=" * 50)
        print("VIDEO PROCESSING STATISTICS")
        print("=" * 50)
        print(f"Frames processed: {stats['frames_processed']}")
        print(f"Total detections: {stats['total_detections']}")
        print(f"Avg detections/frame: {stats['detections_per_frame']:.2f}")
        print(f"Avg processing time: {stats['avg_processing_time']:.1f} ms")
        print(f"Average FPS: {stats['avg_fps']:.1f}")
        print(f"FPS range: {stats['min_fps']:.1f} - {stats['max_fps']:.1f}")
        print(f"\nDetections per class:")
        for class_name, count in stats['detections_per_class'].items():
            print(f"  {class_name}: {count}")
        print("=" * 50)


def process_video(
        model,
        video_path: str,
        output_path: str,
        class_names: List[str],
        conf_threshold: float = 0.25,
        skip_frames: int = 1,
        resize: Optional[Tuple[int, int]] = None,
        show_progress: bool = True,
        save_output: bool = True
) -> Dict:
    """
    Обработка видео с детекцией объектов

    Args:
        model: Модель детекции
        video_path: Путь к входному видео
        output_path: Путь для сохранения результата
        class_names: Имена классов
        conf_threshold: Порог уверенности
        skip_frames: Обрабатывать каждый N-й кадр (1 = все кадры)
        resize: Изменить размер выходного видео (width, height)
        show_progress: Показывать прогресс
        save_output: Сохранять ли видео

    Returns:
        Словарь со статистикой обработки
    """
    cap = None
    out = None
    processor = None

    try:
        # Открываем видео
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        # Информация о видео
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        logger.info(f"Video: {total_frames} frames, {fps} FPS, {orig_width}x{orig_height}")

        # Создаем VideoWriter
        if save_output:
            if resize:
                out_width, out_height = resize
            else:
                out_width, out_height = orig_width, orig_height

            # Создаём папку, если её нет
            output_path_obj = Path(output_path)
            output_path_obj.parent.mkdir(parents=True, exist_ok=True)

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path_obj), fourcc, fps, (out_width, out_height))

            if not out.isOpened():
                raise RuntimeError(f"Failed to create video writer: {output_path}")

        # Создаем процессор
        processor = VideoProcessor(
            model=model,
            class_names=class_names,
            conf_threshold=conf_threshold
        )

        # Обработка кадров
        from tqdm import tqdm

        frame_count = 0
        pbar = tqdm(total=total_frames, desc="Processing video") if show_progress else None

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Пропускаем кадры если нужно
            if frame_count % skip_frames != 0:
                if save_output and out is not None:
                    out.write(frame)
                if pbar:
                    pbar.update(1)
                continue

            # Обрабатываем кадр
            result = processor.process_frame(frame, frame_count)

            # Изменяем размер если нужно
            if resize and save_output:
                output_frame = cv2.resize(result.frame, resize)
            else:
                output_frame = result.frame

            # Сохраняем
            if save_output and out is not None:
                out.write(output_frame)

            if pbar:
                pbar.update(1)

        if pbar:
            pbar.close()

        # Статистика
        stats = processor.get_stats()
        processor.print_stats()

        # Проверяем, что файл сохранился
        if save_output and out is not None:
            output_file = Path(output_path)
            if output_file.exists():
                size_mb = output_file.stat().st_size / (1024 * 1024)
                logger.info(f"✅ Video saved: {output_path} ({size_mb:.2f} MB)")
                print(f"✅ Video saved: {output_path} ({size_mb:.2f} MB)")
            else:
                logger.error(f"❌ Video NOT saved: {output_path}")
        elif not save_output:
            logger.info("ℹ️ Video saving was disabled (save_output=False)")

        return stats

    except Exception as e:
        logger.error(f"Error processing video: {e}")
        raise

    finally:
        # Гарантированно закрываем все ресурсы
        if cap is not None:
            cap.release()
        if out is not None:
            out.release()
            logger.debug(f"Video writer released")
