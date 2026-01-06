from ultralytics import YOLO
import torch
from pathlib import Path

# Get absolute paths
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_PATH = SCRIPT_DIR.parent / "dataset" / "data.yaml"
PROJECT_PATH = SCRIPT_DIR.parent / "models"

model = YOLO('yolov8n.pt')

results = model.train(
    data=str(DATA_PATH),
    epochs=100,              # More epochs
    imgsz=640,
    batch=16,
    device='0' if torch.cuda.is_available() else 'cpu',
    patience=20,             # More patience before early stopping
    save=True,
    plots=True,
    project=str(PROJECT_PATH),
    name='objects_v2',       # New version
    verbose=True,
    augment=True,            # Enable augmentation
    hsv_h=0.015,             # Hue augmentation
    hsv_s=0.7,               # Saturation augmentation  
    hsv_v=0.4,               # Value augmentation
    degrees=10,              # Rotation
    translate=0.1,           # Translation
    scale=0.5,               # Scale
    fliplr=0.5,              # Horizontal flip
    mosaic=1.0,              # Mosaic augmentation
)
print("Training completed. Results:", results)