"""
Train YOLOv11n on NEU-DET steel defect dataset
"""
import sys
from pathlib import Path
from ultralytics import YOLO

# Paths
BASE = Path(__file__).parent.parent
DATA_YAML = BASE / "data/processed/yolo/data.yaml"
OUTPUT_DIR = BASE / "results/models/yolo11n"

def main():
    print("="*60)
    print("YOLOv11n Training on NEU-DET")
    print("="*60)
    
    # Load YOLOv11n pretrained model
    model = YOLO("yolo11n.pt")
    
    # Train
    results = model.train(
        data=str(DATA_YAML),
        epochs=50,
        imgsz=640,
        batch=8,
        patience=10,
        device="cpu",
        project=str(OUTPUT_DIR.parent),
        name=OUTPUT_DIR.name,
        exist_ok=True,
        verbose=True,
        plots=True,
        save=True,
        val=True,
    )
    
    print("\n" + "="*60)
    print(f"Training complete. Best model: {OUTPUT_DIR}/weights/best.pt")
    print("="*60)
    
    # Validate
    metrics = model.val()
    print(f"\nValidation mAP@0.5: {metrics.box.map50:.3f}")
    print(f"Validation mAP@0.5:0.95: {metrics.box.map:.3f}")

if __name__ == "__main__":
    main()
