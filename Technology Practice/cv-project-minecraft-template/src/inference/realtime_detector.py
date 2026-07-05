"""
Детекция в реальном времени с вебкамеры или стрима
"""
from pathlib import Path

import cv2
import numpy as np
import time
from typing import Dict, List, Optional, Callable
import logging
from collections import deque

logger = logging.getLogger(__name__)


class RealtimeDetector:
    """
    Детектор для работы в реальном времени

    Особенности:
    - Асинхронная обработка кадров
    - Автоматический пропуск кадров при низком FPS
    - Визуализация FPS и статистики
    - Горячие клавиши для управления
    """

    def __init__(
            self,
            model,
            class_names: List[str],
            conf_threshold: float = 0.25,
            target_fps: int = 30,
            window_name: str = "Minecraft Mobs Detection"
    ):
        """
        Args:
            model: Модель детекции
            class_names: Имена классов
            conf_threshold: Порог уверенности
            target_fps: Целевой FPS
            window_name: Название окна
        """
        self.model = model
        self.class_names = class_names
        self.conf_threshold = conf_threshold
        self.target_fps = target_fps
        self.window_name = window_name

        # Для FPS
        self.fps_history = deque(maxlen=30)
        self.last_time = time.time()
        self.frame_count = 0

        # Горячие клавиши
        self.paused = False
        self.show_help = True
        self.recording = False
        self.record_writer = None

        # Кэш для предсказаний
        self.last_predictions = None
        self.prediction_ttl = 0  # Как долго показывать старые предсказания

        logger.info(f"RealtimeDetector initialized")
        logger.info(f"Target FPS: {target_fps}, Conf threshold: {conf_threshold}")

    def run(
            self,
            source: int = 0,
            fullscreen: bool = False,
            callback: Optional[Callable] = None
    ):
        """
        Запуск детекции в реальном времени

        Args:
            source: Источник (0 = вебкамера, или путь к файлу/стриму)
            fullscreen: На весь экран
            callback: Функция обратного вызова для каждого кадра
        """
        # Открываем источник
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise ValueError(f"Cannot open source: {source}")

        # Настройка разрешения
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # Создаем окно
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        if fullscreen:
            cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        logger.info("Starting realtime detection. Press 'q' to quit, 'h' for help")

        while True:
            # Обработка клавиш
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):  # Выход
                break
            elif key == ord('p'):  # Пауза
                self.paused = not self.paused
                logger.info(f"{'Paused' if self.paused else 'Resumed'}")
            elif key == ord('h'):  # Помощь
                self.show_help = not self.show_help
            elif key == ord('+') or key == ord('='):  # Увеличить порог
                self.conf_threshold = min(1.0, self.conf_threshold + 0.05)
            elif key == ord('-'):  # Уменьшить порог
                self.conf_threshold = max(0.05, self.conf_threshold - 0.05)
            elif key == ord('r'):  # Запись
                self._toggle_recording()

            if self.paused:
                continue

            # Читаем кадр
            ret, frame = cap.read()
            if not ret:
                break

            self.frame_count += 1

            # Пропускаем кадры если нужно (для поддержания target FPS)
            current_time = time.time()
            if self.last_predictions and (current_time - self.last_time) < 1.0 / self.target_fps:
                # Используем кэшированные предсказания
                frame = self._draw_predictions(frame, self.last_predictions)
            else:
                # Новое предсказание
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                try:
                    predictions = self.model.predict(
                        frame_rgb,
                        conf_threshold=self.conf_threshold
                    )
                    self.last_predictions = predictions
                    self.last_time = current_time

                    # Отрисовка
                    frame = self._draw_predictions(frame, predictions)

                except Exception as e:
                    logger.error(f"Prediction error: {e}")

            # Добавляем оверлей
            frame = self._add_overlay(frame)

            # Вызываем callback если есть
            if callback:
                callback(frame, self.last_predictions)

            # Запись если активна
            if self.recording and self.record_writer:
                self.record_writer.write(frame)

            # Показываем
            cv2.imshow(self.window_name, frame)

        # Очистка
        cap.release()
        if self.record_writer:
            self.record_writer.release()
        cv2.destroyAllWindows()

        logger.info("Realtime detection stopped")

    def _draw_predictions(self, frame: np.ndarray, predictions: Dict) -> np.ndarray:
        """Отрисовка предсказаний"""
        colors = {
            'zombie': (0, 255, 0),
            'skeleton': (255, 255, 255),
            'creeper': (0, 255, 0),
            'spider': (255, 0, 0),
            'enderman': (128, 0, 128),
            'pig': (255, 192, 203)
        }

        for box, score, class_name in zip(
                predictions.get('boxes', []),
                predictions.get('scores', []),
                predictions.get('class_names', [])
        ):
            if score < self.conf_threshold:
                continue

            x1, y1, x2, y2 = map(int, box[:4])
            color = colors.get(class_name, (0, 255, 0))

            # Бокс
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Метка
            label = f"{class_name}: {score:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return frame

    def _add_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Добавление информационного оверлея"""
        # Подсчитываем FPS
        current_time = time.time()
        self.fps_history.append(1.0 / max(0.001, current_time - self.last_time))
        avg_fps = np.mean(self.fps_history)

        # Информационная панель
        h, w = frame.shape[:2]
        overlay = frame.copy()

        # Полупрозрачный фон для статистики
        cv2.rectangle(overlay, (10, 10), (300, 120), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)

        # FPS
        fps_color = (0, 255, 0) if avg_fps >= self.target_fps * 0.8 else (0, 165, 255)
        cv2.putText(frame, f"FPS: {avg_fps:.1f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, fps_color, 2)

        # Порог уверенности
        cv2.putText(frame, f"Conf: {self.conf_threshold:.2f}", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Детекции
        if self.last_predictions:
            num_detections = len(self.last_predictions.get('boxes', []))
            cv2.putText(frame, f"Objects: {num_detections}", (20, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Помощь
        if self.show_help:
            help_text = [
                "Controls:",
                "Q - Quit | P - Pause | H - Help",
                "+/- - Confidence threshold",
                "R - Start/Stop recording"
            ]

            for i, text in enumerate(help_text):
                y = h - 100 + i * 25
                cv2.putText(frame, text, (20, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return frame

    def _toggle_recording(self):
        """Включение/выключение записи"""
        if self.recording:
            if self.record_writer:
                self.record_writer.release()
            self.recording = False
            logger.info("Recording stopped")
        else:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"outputs/videos/recording_{timestamp}.mp4"

            Path("outputs/videos").mkdir(parents=True, exist_ok=True)

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.record_writer = cv2.VideoWriter(filename, fourcc, 30.0, (1280, 720))
            self.recording = True
            logger.info(f"Recording started: {filename}")
