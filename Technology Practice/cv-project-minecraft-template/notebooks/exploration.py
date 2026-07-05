# notebooks/exploration.py
"""Быстрый анализ датасета (без зависаний)"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import yaml
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from tqdm import tqdm
import os

# Загружаем конфиг
with open('configs/default.yaml', 'r') as f:
    config = yaml.safe_load(f)

print("=" * 50)
print("DATASET EXPLORATION")
print("=" * 50)

# Проверяем путь
dataset_path = Path(config['data']['raw_path'])
print(f"\nDataset path: {dataset_path}")
print(f"Path exists: {dataset_path.exists()}")

# Смотрим что внутри
for split in ['train', 'val', 'test']:
    split_dir = dataset_path / split
    if split_dir.exists():
        images_dir = split_dir / 'images'
        labels_dir = split_dir / 'labels'
        if images_dir.exists():
            images = sorted(list(images_dir.glob('*')))
            labels = sorted(list(labels_dir.glob('*'))) if labels_dir.exists() else []
            print(f"\n{split.upper()}:")
            print(f"  Images: {len(images)}")
            print(f"  Labels: {len(labels)}")
            if images:
                print(f"  Examples: {[f.name for f in images[:3]]}")

# Анализ аннотаций напрямую
print("\n\nAnalyzing annotations (first 1000 train images)...")
class_count = Counter()
objects_per_image = []
bad_annotations = 0
total_objects = 0

train_images_dir = dataset_path / 'train' / 'images'
train_labels_dir = dataset_path / 'train' / 'labels'

image_files = sorted(list(train_images_dir.glob('*')))[:1000]

for img_file in tqdm(image_files):
    label_file = train_labels_dir / (img_file.stem + '.txt')
    num_objects = 0

    if label_file.exists():
        try:
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        coords = [float(x) for x in parts[1:5]]

                        # Проверяем валидность координат
                        if all(-0.01 <= c <= 1.01 for c in coords) and 0 <= class_id < len(
                                config['data']['class_names']):
                            class_name = config['data']['class_names'][class_id]
                            class_count[class_name] += 1
                            num_objects += 1
                            total_objects += 1
                        else:
                            bad_annotations += 1
        except Exception:
            bad_annotations += 1

    objects_per_image.append(num_objects)

print(f"\nBad annotations skipped: {bad_annotations}")
print(f"Total valid objects: {total_objects}")
print(f"Avg objects per image: {total_objects / len(image_files):.1f}")

print("\nClass distribution:")
for name, count in class_count.most_common():
    print(f"  {name:12s}: {count:5d} ({count / total_objects * 100:.1f}%)")

# Проверка датасета через загрузчик (одно изображение)
print("\n\nTesting dataset loader...")
from src.dataset import MinecraftMobsDataset

try:
    ds = MinecraftMobsDataset(
        data_dir=str(dataset_path),
        split='train',
        img_size=320,
        class_names=config['data']['class_names']
    )
    print(f"Dataset size: {len(ds)}")

    # Пробуем загрузить несколько изображений
    success = 0
    for i in range(10):
        try:
            sample = ds[i]
            success += 1
        except Exception:
            pass

    print(f"Successfully loaded: {success}/10 images")

    if success > 0:
        sample = ds[0]
        print(f"Image shape: {sample['image'].shape}")
        print(f"Boxes: {sample['boxes'].shape[0]} objects")
except Exception as e:
    print(f"Dataset loader error: {e}")

# Графики
os.makedirs('results/plots', exist_ok=True)

# 1. Распределение классов
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

names = [n for n, _ in class_count.most_common()]
counts = [c for _, c in class_count.most_common()]
colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(names)))

axes[0].bar(names, counts, color=colors, edgecolor='black')
axes[0].set_title('Class Distribution (first 1000 images)', fontweight='bold')
axes[0].set_xlabel('Class')
axes[0].set_ylabel('Count')
axes[0].tick_params(axis='x', rotation=45)

# Добавляем проценты на столбцы
for i, (name, count) in enumerate(zip(names, counts)):
    pct = count / total_objects * 100
    axes[0].text(i, count + 5, f'{pct:.1f}%', ha='center', fontsize=9)

# 2. Распределение объектов на изображении
axes[1].hist(objects_per_image, bins=range(0, max(objects_per_image) + 2),
             color='steelblue', edgecolor='black', alpha=0.7)
axes[1].set_title('Objects per Image', fontweight='bold')
axes[1].set_xlabel('Number of objects')
axes[1].set_ylabel('Number of images')
axes[1].axvline(np.mean(objects_per_image), color='red', linestyle='--',
                label=f'Mean: {np.mean(objects_per_image):.1f}')
axes[1].legend()
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('results/plots/class_distribution.png', dpi=150, bbox_inches='tight')
print("\nPlot saved to results/plots/class_distribution.png")

# 3. Сохраняем статистику
import json

stats = {
    'train_images': len(image_files),
    'total_objects': total_objects,
    'avg_objects_per_image': round(total_objects / len(image_files), 1),
    'bad_annotations': bad_annotations,
    'class_distribution': dict(class_count),
    'class_names': config['data']['class_names']
}

with open('results/dataset_statistics.json', 'w') as f:
    json.dump(stats, f, indent=2)
print("Statistics saved to results/dataset_statistics.json")
print("Exploration complete!")