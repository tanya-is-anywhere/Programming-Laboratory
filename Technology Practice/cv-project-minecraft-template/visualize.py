# ====== 
# visualize.py - файл, с помощью которого можно обработать картинку: рамки вокруг объектов рисуются на основе предсказания модели
# ======
import cv2
import matplotlib.pyplot as plt
from src.dataset.dataset import MinecraftMobsDataset
import torch
import numpy as np

dataset = MinecraftMobsDataset(
    data_dir='D:/otherfiles/learning_practice_technology/minecraft-mobs-yolo-dataset/minecraft_mobs_yolo',
    split='train',
    img_size=640,
    box_format='xyxy'
)
def denormalize(image_tensor):
    """
    Превращает нормализованный тензор (C, H, W) обратно в numpy (H, W, C)
    с естественными цветами (значения 0..255)
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    image = image_tensor * std + mean
    image = torch.clamp(image, 0, 1)
    image = image.permute(1, 2, 0).numpy()
    image = (image * 255).astype(np.uint8)
    return image

sample = dataset[2]  # то самое изображение с 2 мобами
image = sample['pixel_values'].permute(1, 2, 0).numpy()
image_tensor = sample['pixel_values']
boxes = sample['labels']['boxes'].numpy().astype(int)

# Денормализация для отображения
image = (image * 255).astype('uint8')
image_restored = denormalize(image_tensor)
image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

def draw_boxes(image, boxes, color=(0, 255, 0)):
    img_copy = image.copy()
    for (x1, y1, x2, y2) in boxes:
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)
    return img_copy

image_normalized_with_boxes = draw_boxes(image, boxes)
image_restored_with_boxes = draw_boxes(image_restored, boxes)
combined = np.hstack([image_normalized_with_boxes, image_restored_with_boxes])
plt.figure(figsize=(12, 8))
plt.imshow(combined)
plt.title("Слева: Нормализованное изображение (что видит нейросеть) | Справа: После денормализации (естественные цвета)")
plt.axis('off')
plt.show()
combined_bgr = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
cv2.imwrite('comparison_normalization.png', combined_bgr)
