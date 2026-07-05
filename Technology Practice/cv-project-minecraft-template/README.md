# Minecraft Mobs Detection — Сравнение моделей компьютерного зрения

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue" alt="Python"/>
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-orange" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/Ultralytics-8.0%2B-red" alt="Ultralytics"/>
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License"/>
</div>

---

##  О проекте

Данный проект представляет собой сравнительное исследование пяти архитектур детекции объектов для обнаружения мобов в игре **Minecraft**. В рамках работы были обучены и протестированы модели:

- **YOLO** (v8n) — классический одностадийный детектор
- **YOLO-World** — open-vocabulary детектор с текстовыми промптами
- **Faster R-CNN** (MobileNetV3) — двухстадийный детектор
- **RT-DETR** — трансформерный детектор реального времени
- **DETR** — классический трансформерный детектор (экспериментальный)

**Ключевой результат:** модель **YOLO-World** показала наилучшую точность (mAP@0.5 = 0.928), а **YOLO** — оптимальное соотношение скорости и качества (74 FPS).

---

##  Основные возможности

-  **Обучение 5 моделей** через единый интерфейс
-  **Сравнительный анализ** метрик (mAP, Precision, Recall, F1)
-  **Обработка видео** с детекцией в реальном времени
-  **Визуализация** результатов (графики, матрица ошибок, рамки на изображениях)
-  **Гибкая настройка** через YAML-конфиг и аргументы командной строки
-  **Модульная архитектура** для лёгкого расширения

---

##  Структура проекта
```
cv-project-minecraft-template/
├── configs/
│ └── default.yaml # Конфигурация обучения
├── data/
│ ├── raw/ # Сырые данные (датасет)
│ └── processed/ # Предобработанные данные
├── results/
│ ├── logs/ # Логи экспериментов
│ ├── plots/ # Графики и визуализации
│ └── predictions/ # Результаты детекции
├── outputs/
│ └── videos/ # Выходные видео
├── scripts/
│ ├── detect_video.py # Обработка видео
│ └── demo_realtime.py # Демо с веб-камерой
├── src/
│ ├── dataset/ # Классы датасетов
│ ├── models/ # Реализации моделей
│ ├── training/ # Циклы обучения
│ ├── inference/ # Инференс видео
│ └── utils/ # Вспомогательные функции
├── main.py # Главная точка входа
├── requirements.txt # Зависимости
└── README.md # Этот файл
```

---

## Установка и настройка

### 1. Клонируй репозиторий

```bash
git clone https://github.com/your-username/cv-project-minecraft-template.git
cd cv-project-minecraft-template