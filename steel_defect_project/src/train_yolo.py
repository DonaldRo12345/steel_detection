"""
YOLO model training script for steel surface defect detection.
Uses Ultralytics YOLOv8 implementation.
"""

import os
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
import torch
from ultralytics import YOLO

from utils import set_seed, get_device, save_environment_info, save_metrics, setup_logging


logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train YOLO model for steel defect detection')
    
    parser.add_argument('--data', type=str, required=True,
                       help='Path to data.yaml file')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config file (optional)')
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                       help='YOLO model variant (yolov8n, yolov8s, yolov8m, etc.)')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--img-size', type=int, default=640,
                       help='Image size')
    parser.add_argument('--lr', type=float, default=0.01,
                       help='Learning rate')
    parser.add_argument('--device', type=str, default='',
                       help='Device (cuda:0, cpu, etc.)')
    parser.add_argument('--workers', type=int, default=8,
                       help='Number of workers for data loading')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--output-dir', type=str, default='results/models',
                       help='Output directory for models')
    parser.add_argument('--name', type=str, default='yolo_steel',
                       help='Experiment name')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint')
    parser.add_argument('--pretrained', action='store_true',
                       help='Use pretrained weights')
    
    return parser.parse_args()


def train_yolo(args):
    """
    Train YOLO model.
    
    Args:
        args: Command line arguments
    """
    # Setup
    set_seed(args.seed)
    setup_logging('experiments/logs', 'yolo_training.log')
    
    logger.info("="*80)
    logger.info("YOLO Training for Steel Surface Defect Detection")
    logger.info("="*80)
    
    # Save environment info
    env_file = Path(args.output_dir) / 'env.json'
    env_info = save_environment_info(str(env_file))
    logger.info(f"Environment: {json.dumps(env_info, indent=2)}")
    
    # Device
    device = args.device if args.device else ('0' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load model
    logger.info(f"Loading model: {args.model}")
    
    if args.resume:
        model = YOLO(args.resume)
        logger.info(f"Resuming from checkpoint: {args.resume}")
    else:
        model = YOLO(args.model)
        logger.info(f"Using {'pretrained' if args.pretrained else 'random'} weights")
    
    # Training arguments
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
        'optimizer': 'SGD',
        'verbose': True,
        'seed': args.seed,
        'deterministic': True,
        'plots': True,
        'save': True,
        'save_period': 10,  # Save checkpoint every 10 epochs
        'val': True,
    }
    
    logger.info(f"Training arguments: {json.dumps(train_args, indent=2, default=str)}")
    
    # Train
    try:
        logger.info("Starting training...")
        results = model.train(**train_args)
        logger.info("Training completed successfully!")
        
        # Save best model to standard location
        best_model_path = Path(args.output_dir) / args.name / 'weights' / 'best.pt'
        output_best = Path(args.output_dir) / 'yolo_best.pt'
        
        if best_model_path.exists():
            import shutil
            shutil.copy(best_model_path, output_best)
            logger.info(f"Best model saved to {output_best}")
        
        # Extract and save metrics
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
        
        # Try to extract final metrics from results
        if hasattr(results, 'results_dict'):
            metrics.update(results.results_dict)
        
        metrics_file = Path(args.output_dir).parent / 'metrics' / 'yolo_training.json'
        save_metrics(metrics, str(metrics_file))
        
        logger.info(f"Training metrics saved to {metrics_file}")
        logger.info("="*80)
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        raise


def main():
    """Main entry point."""
    args = parse_args()
    train_yolo(args)


if __name__ == '__main__':
    main()
