import kagglehub
from ultralytics import YOLO

# ========================================
# Dataset Download (Kaggle)
# ========================================
dataset_path = kagglehub.dataset_download("linaaabrahim/dataset1")
print("📁 Dataset path:", dataset_path)


# ========================================
# YOLOv8 Training — Sunflower Seed Detection
# Fertilized / Unfertilized Classification
# ========================================

# Load pretrained YOLOv8 model
model = YOLO("yolov8m.pt")

# Train model
results = model.train(
    data=f"{dataset_path}/data.yaml",

    # Core training settings
    epochs=100,
    imgsz=1280,
    batch=1,          # Kaggle GPU constraint
    device=0,

    optimizer="AdamW",

    # Detection settings
    conf=0.05,
    iou=0.3,
    max_det=3000,

    # Augmentation
    mosaic=0.1,
    hsv_h=0.02,
    hsv_s=0.3,
    hsv_v=0.3,
    translate=0.05,
    scale=0.15,

    # Learning rate
    lr0=0.0002,
    lrf=0.01,
    momentum=0.97,
    weight_decay=0.0005,

    # Stability
    patience=50,
    workers=2,

    # Experiment name
    name="sunflower_seed_detection_v1",
    verbose=True
)

print("\n✅ Training completed successfully!")
