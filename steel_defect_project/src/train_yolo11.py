"""
Train YOLOv11n on NEU-DET steel defect dataset
"""
import json
import shutil
import argparse
from pathlib import Path

import torch
from ultralytics import YOLO

# Paths
BASE = Path(__file__).parent.parent
DATA_YAML = BASE / "data/processed/yolo/data.yaml"


def parse_args():
    p = argparse.ArgumentParser(description='Train YOLOv11n for steel defect detection')
    p.add_argument('--data', type=str, default=str(DATA_YAML),
                   help='Path to data.yaml')
    p.add_argument('--model', type=str, default='yolo11n.pt',
                   help='Base model')
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--img-size', type=int, default=640)
    p.add_argument('--device', type=str, default='',
                   help="Device ('0' for GPU, 'cpu'); auto-detected if empty")
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--patience', type=int, default=10)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output-dir', type=str, default=str(BASE / 'results/models'))
    p.add_argument('--name', type=str, default='yolo11n')
    return p.parse_args()


def train_yolo11(args):
    # GPU si disponible (serveur), sinon CPU (poste local)
    device = args.device if args.device else ('0' if torch.cuda.is_available() else 'cpu')

    print("=" * 60)
    print("YOLOv11n Training on NEU-DET")
    print("=" * 60)
    print(f"Data   : {args.data}")
    print(f"Device : {device}")
    print(f"Epochs : {args.epochs} | batch {args.batch_size} | imgsz {args.img_size}")
    print(f"Run    : {Path(args.output_dir) / args.name}")

    # Load YOLOv11n pretrained model
    model = YOLO(args.model)

    # Train
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.img_size,
        batch=args.batch_size,
        patience=args.patience,
        device=device,
        workers=args.workers,
        project=args.output_dir,
        name=args.name,
        exist_ok=True,
        verbose=True,
        plots=True,
        save=True,
        val=True,
        seed=args.seed,
        deterministic=True,
    )

    run_dir = Path(args.output_dir) / args.name
    print("\n" + "=" * 60)
    print(f"Training complete. Best model: {run_dir}/weights/best.pt")
    print("=" * 60)

    # Flat copy under the name expected by app.py / Dockerfile.web
    best_path = run_dir / 'weights' / 'best.pt'
    output_best = Path(args.output_dir) / 'yolo11n_best.pt'
    if best_path.exists():
        shutil.copy(best_path, output_best)
        print(f"Best model -> {output_best}")

    # Validate
    metrics = model.val()
    print(f"\nValidation mAP@0.5: {metrics.box.map50:.3f}")
    print(f"Validation mAP@0.5:0.95: {metrics.box.map:.3f}")

    summary = {
        'model': args.model,
        'data': args.data,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'img_size': args.img_size,
        'device': str(device),
        'seed': args.seed,
        'training_complete': True,
        'best_model_path': str(output_best) if output_best.exists() else None,
        'mAP_50': float(metrics.box.map50),
        'mAP_50_95': float(metrics.box.map),
    }
    metrics_file = Path(args.output_dir).parent / 'metrics' / 'yolo11n_training.json'
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    metrics_file.write_text(json.dumps(summary, indent=2))
    print(f"Metrics -> {metrics_file}")


def main():
    train_yolo11(parse_args())


if __name__ == "__main__":
    main()
