"""
visualisation.py - Визуализация результатов обучения
"""

import json
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import seaborn as sns
import cv2
from torchvision import transforms as T
import torch
import numpy as np

def plot_training_history(
    history_path: str,
    save_dir: str = 'results/plots',
    model_name: str = None
):
    """
    Строит графики обучения: loss и mAP по эпохам.

    Args:
        history_path: Путь к history.json или results.csv
        save_dir: Папка для сохранения графиков
        model_name: Имя модели (для подписи)
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Определяем формат файла
    history_path = Path(history_path)
    if history_path.suffix == '.json':
        with open(history_path, 'r') as f:
            history = json.load(f)
        is_csv = False
    else:
        df = pd.read_csv(history_path)
        is_csv = True

    if model_name is None:
        model_name = history_path.parent.name

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # === График 1: Loss ===
    ax = axes[0]
    if is_csv:
        # Для YOLO (results.csv)
        if 'train/box_loss' in df:
            ax.plot(df['train/box_loss'], label='Box Loss', marker='o', markersize=3)
        if 'train/cls_loss' in df:
            ax.plot(df['train/cls_loss'], label='Cls Loss', marker='s', markersize=3)
        if 'val/box_loss' in df:
            ax.plot(df['val/box_loss'], label='Val Box Loss', marker='^', markersize=3)
    else:
        # Для DETR / Faster R-CNN (history.json)
        if 'train_loss' in history:
            ax.plot(history['train_loss'], label='Train Loss', marker='o', markersize=3)
        if 'val_loss' in history:
            ax.plot(history['val_loss'], label='Val Loss', marker='s', markersize=3)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(f'{model_name} - Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # === График 2: Метрики ===
    ax = axes[1]
    if is_csv:
        # Для YOLO
        if 'metrics/mAP_0.5' in df:
            ax.plot(df['metrics/mAP_0.5'], label='mAP@0.5', marker='o', markersize=3)
        if 'metrics/mAP_0.5_0.95' in df:
            ax.plot(df['metrics/mAP_0.5_0.95'], label='mAP@0.5:0.95', marker='s', markersize=3)
        if 'metrics/precision' in df:
            ax.plot(df['metrics/precision'], label='Precision', marker='^', markersize=3)
        if 'metrics/recall' in df:
            ax.plot(df['metrics/recall'], label='Recall', marker='v', markersize=3)
    else:
        # Для DETR / Faster R-CNN
        if 'mAP_0.5' in history and any(v != 0 for v in history['mAP_0.5']):
            ax.plot(history['mAP_0.5'], label='mAP@0.5', marker='o', markersize=3)
        if 'mAP_0.5_0.95' in history and any(v != 0 for v in history['mAP_0.5_0.95']):
            ax.plot(history['mAP_0.5_0.95'], label='mAP@0.5:0.95', marker='s', markersize=3)
        if 'precision' in history and any(v != 0 for v in history['precision']):
            ax.plot(history['precision'], label='Precision', marker='^', markersize=3)
        if 'recall' in history and any(v != 0 for v in history['recall']):
            ax.plot(history['recall'], label='Recall', marker='v', markersize=3)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Metric')
    ax.set_title(f'{model_name} - Metrics')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = save_dir / f'{model_name}_training_history.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Графики сохранены: {save_path}")
    return str(save_path)

def plot_confusion_matrix(
    cm_path: str = None,
    cm_data: np.ndarray = None,
    class_names: list = None,
    save_dir: str = 'results/plots',
    model_name: str = 'model'
):
    """
    Строит нормализованную матрицу ошибок.

    Args:
        cm_path: Путь к confusion_matrix.json (из Ultralytics)
        cm_data: Матрица ошибок (если передаётся напрямую)
        class_names: Список имён классов
        save_dir: Папка для сохранения
        model_name: Имя модели
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Загружаем матрицу
    if cm_data is None and cm_path is not None:
        cm_path = Path(cm_path)
        if cm_path.suffix == '.json':
            with open(cm_path, 'r') as f:
                data = json.load(f)
                cm_data = np.array(data['confusion_matrix'])
                if class_names is None:
                    class_names = data.get('class_names', [f'Class_{i}' for i in range(cm_data.shape[0])])
        else:
            # Если файл .npy
            cm_data = np.load(cm_path)

    if cm_data is None:
        print("⚠️ Нет данных для матрицы ошибок")
        return None

    # Нормализация
    cm_norm = cm_data.astype('float') / cm_data.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)

    if class_names is None:
        class_names = [f'Class_{i}' for i in range(cm_data.shape[0])]

    # Добавляем background, если его нет
    if len(class_names) < cm_data.shape[0]:
        class_names = class_names + ['background']

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        square=True,
        cbar_kws={'label': 'Normalized Frequency'}
    )

    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f'Confusion Matrix - {model_name}')

    plt.tight_layout()
    save_path = save_dir / f'confusion_matrix_{model_name}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Матрица ошибок сохранена: {save_path}")
    return str(save_path)

def visualize_detections(
    model,
    image_path: str,
    save_dir: str = 'results/predictions',
    conf_threshold: float = 0.25,
    model_name: str = 'model',
    processor=None,
    img_size: int = 640
):
    """Визуализирует детекции на одном изображении"""

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(image_path))
    if image is None:
        print(f"❌ Не удалось загрузить изображение: {image_path}")
        return None

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = image.shape[:2]

    boxes, scores, labels = [], [], []

    try:
        # --- Faster R-CNN (torchvision) ---
        if hasattr(model, 'roi_heads') or hasattr(model, 'model') and hasattr(model.model, 'roi_heads'):
            transform = T.Compose([
                T.ToTensor(),
                T.Resize((img_size, img_size)),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            img_tensor = transform(image_rgb).unsqueeze(0)
            model.eval()
            with torch.no_grad():
                outputs = model(img_tensor)[0]

            boxes = outputs['boxes'].cpu().numpy()
            scores = outputs['scores'].cpu().numpy()
            labels = outputs['labels'].cpu().numpy()

            # Масштабируем боксы к оригинальному размеру
            scale_w = orig_w / img_size
            scale_h = orig_h / img_size
            boxes = boxes * np.array([scale_w, scale_h, scale_w, scale_h])

        # --- YOLO / Ultralytics ---
        elif hasattr(model, 'predict'):
            results = model.predict(image_path, conf=conf_threshold)
            if len(results) > 0 and hasattr(results[0], 'boxes'):
                boxes = results[0].boxes.xyxy.cpu().numpy()
                scores = results[0].boxes.conf.cpu().numpy()
                labels = results[0].boxes.cls.cpu().numpy().astype(int)

        # --- DETR ---
        elif processor is not None:
            from PIL import Image
            image_pil = Image.fromarray(image_rgb)
            inputs = processor(images=image_pil, return_tensors="pt")
            outputs = model(**inputs)
            target_sizes = torch.tensor([image_pil.size[::-1]])
            results = processor.post_process_object_detection(
                outputs, target_sizes=target_sizes, threshold=conf_threshold
            )[0]
            boxes = results['boxes'].detach().cpu().numpy()
            scores = results['scores'].detach().cpu().numpy()
            labels = results['labels'].detach().cpu().numpy()

    except Exception as e:
        print(f"⚠️ Ошибка при инференсе: {e}")
        return None

    # Фильтруем по порогу
    if len(scores) > 0:
        mask = scores > conf_threshold
        boxes = boxes[mask] if len(boxes) > 0 else []
        scores = scores[mask] if len(scores) > 0 else []
        labels = labels[mask] if len(labels) > 0 else []

    class_names = ['creeper', 'skeleton', 'spider', 'zombie', 'enderman']
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]

    image_cv = image.copy()
    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = map(int, box)
        color = colors[label % len(colors)]
        cv2.rectangle(image_cv, (x1, y1), (x2, y2), color, 2)
        text = f'{class_names[label]}: {score:.2f}'
        cv2.putText(image_cv, text, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    save_path = save_dir / f'detection_{model_name}_{Path(image_path).stem}.png'
    cv2.imwrite(str(save_path), image_cv)

    print(f"✅ Детекция сохранена: {save_path}")
    return str(save_path)