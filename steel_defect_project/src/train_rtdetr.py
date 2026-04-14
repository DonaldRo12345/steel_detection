"""
RT-DETR training script for steel surface defect detection.
Uses Ultralytics' RT-DETR implementation (rtdetr-l / rtdetr-x).
RT-DETR is a real-time DETR variant proposed by Baidu (Zhao et al., 2023).
"""

import os
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime

import torch
from ultralytics import RTDETR

from utils import set_seed, get_device, save_environment_info, save_metrics, setup_logging


logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description='Train RT-DETR for steel defect detection')
    p.add_argument('--data', type=str, required=True,
                   help='Path to data.yaml (YOLO format)')
    p.add_argument('--model', type=str, default='rtdetr-l.pt',
                   help='RT-DETR variant: rtdetr-l.pt or rtdetr-x.pt')
    p.add_argument('--epochs', type=int, default=80)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--img-size', type=int, default=640)
    p.add_argument('--lr', type=float, default=0.0001)
    p.add_argument('--device', type=str, default='')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output-dir', type=str, default='results/models')
    p.add_argument('--name', type=str, default='rtdetr_steel')
    p.add_argument('--resume', type=str, default=None)
    p.add_argument('--pretrained', action='store_true', default=True)
    return p.parse_args()


def train_rtdetr(args):
    set_seed(args.seed)
    setup_logging('experiments/logs', 'rtdetr_training.log')

    logger.info("=" * 80)
    logger.info("RT-DETR Training for Steel Surface Defect Detection")
    logger.info("=" * 80)

    env_file = Path(args.output_dir) / 'env.json'
    save_environment_info(str(env_file))

    device = args.device if args.device else ('0' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")

    # Load model
    if args.resume:
        model = RTDETR(args.resume)
        logger.info(f"Resuming from {args.resume}")
    else:
        model = RTDETR(args.model)
        logger.info(f"Loaded {args.model} ({'pretrained' if args.pretrained else 'scratch'})")

    train_args = {
        'data': args.data,
        'epochs': args.epochs,
        'batch': args.batch_size,
        'imgsz': args.img_size,
        'lr0': args.lr,
        'device': device,
        'workers': args.workers,
        'project': args.output_dir,
        'name': args.name,
        'exist_ok': True,
        'pretrained': args.pretrained,
        'optimizer': 'AdamW',
        'verbose': True,
        'seed': args.seed,
        'deterministic': True,
        'plots': True,
        'save': True,
        'save_period': 10,
        'val': True,
    }

    logger.info(f"Train args: {json.dumps(train_args, indent=2, default=str)}")

    try:
        logger.info("Starting RT-DETR training...")
        results = model.train(**train_args)
        logger.info("Training completed!")

        import shutil
        best_path = Path(args.output_dir) / args.name / 'weights' / 'best.pt'
        output_best = Path(args.output_dir) / 'rtdetr_best.pt'
        if best_path.exists():
            shutil.copy(best_path, output_best)
            logger.info(f"Best model → {output_best}")

        metrics = {
            'model': args.model,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'img_size': args.img_size,
            'learning_rate': args.lr,
            'device': str(device),
            'seed': args.seed,
            'training_complete': True,
            'best_model_path': str(output_best) if output_best.exists() else None,
        }
        if hasattr(results, 'results_dict'):
            metrics.update(results.results_dict)

        metrics_file = Path(args.output_dir).parent / 'metrics' / 'rtdetr_training.json'
        save_metrics(metrics, str(metrics_file))
        logger.info(f"Metrics → {metrics_file}")

    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise


def main():
    args = parse_args()
    train_rtdetr(args)


if __name__ == '__main__':
    main()
