from ultralytics import YOLO
import torch

def create_yolo_model(config):
    """Создаёт и настраивает модель YOLOv8."""
    model = YOLO(config['variant'] + '.pt')  # например, yolov8n.pt
    return model