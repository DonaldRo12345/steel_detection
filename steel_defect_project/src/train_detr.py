"""
DETR/RT-DETR model training script for steel surface defect detection.
Implements a lightweight DETR-based detector.
"""

import os
import json
import argparse
import logging
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import cv2
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment

from utils import (set_seed, get_device, save_environment_info, save_metrics, 
                   setup_logging, AverageMeter, format_time, count_parameters, get_lr)


logger = logging.getLogger(__name__)


class DETRLite(nn.Module):
    """Lightweight DETR-like model for object detection."""
    
    def __init__(self, num_classes: int = 6, num_queries: int = 100, 
                 hidden_dim: int = 256, nheads: int = 8, 
                 num_encoder_layers: int = 3, num_decoder_layers: int = 3):
        """
        Initialize DETR-Lite model.
        
        Args:
            num_classes: Number of object classes
            num_queries: Number of object queries
            hidden_dim: Dimension of hidden layers
            nheads: Number of attention heads
            num_encoder_layers: Number of transformer encoder layers
            num_decoder_layers: Number of transformer decoder layers
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.num_queries = num_queries
        
        # Backbone (simple CNN)
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
            
            self._make_layer(64, 128, 2),
            self._make_layer(128, 256, 2),
            self._make_layer(256, hidden_dim, 2),
        )
        
        # Positional encoding
        self.pos_encoder = nn.Parameter(torch.randn(1, hidden_dim, 20, 20))
        
        # Transformer
        self.transformer = nn.Transformer(
            d_model=hidden_dim,
            nhead=nheads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        
        # Object queries
        self.query_embed = nn.Parameter(torch.randn(num_queries, hidden_dim))
        
        # Prediction heads
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)  # +1 for no-object
        self.bbox_embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4)
        )
        
    def _make_layer(self, in_channels, out_channels, num_blocks):
        """Create a conv layer block."""
        layers = []
        for i in range(num_blocks):
            layers.extend([
                nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ])
        layers.append(nn.MaxPool2d(2))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input images [B, 3, H, W]
            
        Returns:
            Dictionary with 'pred_logits' and 'pred_boxes'
        """
        # Backbone
        features = self.backbone(x)  # [B, C, H', W']
        
        # Add positional encoding
        b, c, h, w = features.shape
        pos = self.pos_encoder[:, :, :h, :w]
        features = features + pos
        
        # Flatten spatial dimensions
        features_flat = features.flatten(2).permute(0, 2, 1)  # [B, H'*W', C]
        
        # Expand queries for batch
        queries = self.query_embed.unsqueeze(0).expand(b, -1, -1)  # [B, num_queries, C]
        
        # Transformer
        hs = self.transformer(features_flat, queries)  # [B, num_queries, C]
        
        # Predictions
        outputs_class = self.class_embed(hs)  # [B, num_queries, num_classes+1]
        outputs_coord = self.bbox_embed(hs).sigmoid()  # [B, num_queries, 4]
        
        return {
            'pred_logits': outputs_class,
            'pred_boxes': outputs_coord
        }


# ── Class mapping ─────────────────────────────────────────────────────────────
CLASS_NAMES_DETR = ['crazing', 'inclusion', 'patches', 'pitted_surface',
                    'rolled-in_scale', 'scratches']
CLASS_TO_IDX_DETR = {c: i for i, c in enumerate(CLASS_NAMES_DETR)}


class SteelDETRDataset(Dataset):
    """Steel defect dataset for DETR-Lite.

    Reads from splits.json image paths and VOC-XML annotations.
    Returns normalized (cx, cy, w, h) boxes in [0, 1] as required by DETR.
    """

    def __init__(self, image_paths: List[str], annotations_dir: str, img_size: int = 640):
        self.image_paths = image_paths
        self.ann_dir = Path(annotations_dir)
        self.img_size = img_size

    def __len__(self):
        return len(self.image_paths)

    def _parse_xml(self, xml_path: Path, img_w: int, img_h: int):
        """Parse VOC XML → (labels list, boxes list in cxcywh normalised)."""
        try:
            tree = ET.parse(xml_path)
        except Exception:
            return [], []
        root = tree.getroot()
        labels, boxes = [], []
        for obj in root.findall('object'):
            name = obj.find('name').text.lower().strip()
            if name not in CLASS_TO_IDX_DETR:
                continue
            bndbox = obj.find('bndbox')
            xmin = float(bndbox.find('xmin').text)
            ymin = float(bndbox.find('ymin').text)
            xmax = float(bndbox.find('xmax').text)
            ymax = float(bndbox.find('ymax').text)
            cx = ((xmin + xmax) / 2) / img_w
            cy = ((ymin + ymax) / 2) / img_h
            w  = (xmax - xmin) / img_w
            h  = (ymax - ymin) / img_h
            labels.append(CLASS_TO_IDX_DETR[name])
            boxes.append([cx, cy, w, h])
        return labels, boxes

    def __getitem__(self, idx):
        img_path = Path(self.image_paths[idx])
        img = cv2.imread(str(img_path))
        if img is None:
            img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img_h, img_w = img.shape[:2]
        img_tensor = torch.from_numpy(
            img[:, :, ::-1].transpose(2, 0, 1).copy()).float() / 255.0

        # Derive annotation filename from image stem
        stem = img_path.stem  # e.g. "crazing_1"
        xml_path = self.ann_dir / f'{stem}.xml'
        orig_w, orig_h = img_w, img_h  # after resize both equal img_size
        labels, boxes = self._parse_xml(xml_path, orig_w, orig_h)

        if labels:
            target = {
                'labels': torch.tensor(labels, dtype=torch.long),
                'boxes': torch.tensor(boxes, dtype=torch.float32),
            }
        else:
            target = {
                'labels': torch.zeros(0, dtype=torch.long),
                'boxes': torch.zeros((0, 4), dtype=torch.float32),
            }
        return img_tensor, target


def collate_fn_detr(batch):
    images = torch.stack([b[0] for b in batch])
    targets = [b[1] for b in batch]
    return images, targets


def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """GIoU between two sets of cxcywh boxes (0-1 normalised)."""
    # Convert cxcywh → xyxy
    b1 = torch.stack([
        boxes1[:, 0] - boxes1[:, 2] / 2,
        boxes1[:, 1] - boxes1[:, 3] / 2,
        boxes1[:, 0] + boxes1[:, 2] / 2,
        boxes1[:, 1] + boxes1[:, 3] / 2,
    ], dim=1)
    b2 = torch.stack([
        boxes2[:, 0] - boxes2[:, 2] / 2,
        boxes2[:, 1] - boxes2[:, 3] / 2,
        boxes2[:, 0] + boxes2[:, 2] / 2,
        boxes2[:, 1] + boxes2[:, 3] / 2,
    ], dim=1)
    inter_x1 = torch.max(b1[:, None, 0], b2[None, :, 0])
    inter_y1 = torch.max(b1[:, None, 1], b2[None, :, 1])
    inter_x2 = torch.min(b1[:, None, 2], b2[None, :, 2])
    inter_y2 = torch.min(b1[:, None, 3], b2[None, :, 3])
    inter = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
    area1 = (b1[:, 2] - b1[:, 0]) * (b1[:, 3] - b1[:, 1])
    area2 = (b2[:, 2] - b2[:, 0]) * (b2[:, 3] - b2[:, 1])
    union = area1[:, None] + area2[None, :] - inter
    iou = inter / union.clamp(min=1e-6)
    enc_x1 = torch.min(b1[:, None, 0], b2[None, :, 0])
    enc_y1 = torch.min(b1[:, None, 1], b2[None, :, 1])
    enc_x2 = torch.max(b1[:, None, 2], b2[None, :, 2])
    enc_y2 = torch.max(b1[:, None, 3], b2[None, :, 3])
    enc_area = (enc_x2 - enc_x1).clamp(0) * (enc_y2 - enc_y1).clamp(0)
    giou = iou - (enc_area - union) / enc_area.clamp(min=1e-6)
    return giou


def build_criterion(num_classes: int):
    """Build DETR loss with Hungarian matching."""

    class DETRLoss(nn.Module):
        def __init__(self, num_classes, weight_dict=None):
            super().__init__()
            self.num_classes = num_classes
            self.weight_dict = weight_dict or {'loss_ce': 1.0, 'loss_bbox': 5.0, 'loss_giou': 2.0}
            # +1 for the background (no-object) class
            self.no_object_weight = 0.1

        def _match(self, pred_logits_i, pred_boxes_i, tgt_labels, tgt_boxes):
            """Hungarian matching for a single image.

            pred_logits_i: [num_queries, num_classes+1]
            pred_boxes_i:  [num_queries, 4]  (cxcywh, normalised)
            tgt_labels:    [num_gt]
            tgt_boxes:     [num_gt, 4]
            Returns: (query_indices, gt_indices)
            """
            num_gt = len(tgt_labels)
            if num_gt == 0:
                return torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)

            with torch.no_grad():
                probs = pred_logits_i.softmax(-1)  # [Q, C+1]
                cost_cls = -probs[:, tgt_labels]   # [Q, num_gt]
                giou = generalized_box_iou(pred_boxes_i, tgt_boxes)  # [Q, num_gt]
                cost_bbox = torch.cdist(pred_boxes_i.float(),
                                        tgt_boxes.float(), p=1)      # [Q, num_gt]
                cost = (self.weight_dict['loss_ce'] * cost_cls
                        + self.weight_dict['loss_bbox'] * cost_bbox
                        - self.weight_dict['loss_giou'] * giou)
                row_ind, col_ind = linear_sum_assignment(cost.cpu().numpy())
            return (torch.as_tensor(row_ind, dtype=torch.long),
                    torch.as_tensor(col_ind, dtype=torch.long))

        def forward(self, outputs, targets):
            pred_logits = outputs['pred_logits']   # [B, Q, C+1]
            pred_boxes  = outputs['pred_boxes']    # [B, Q, 4]
            device = pred_logits.device

            total_ce = torch.tensor(0.0, device=device)
            total_bbox = torch.tensor(0.0, device=device)
            total_giou = torch.tensor(0.0, device=device)
            num_matched = 0

            for i, tgt in enumerate(targets):
                tgt_labels = tgt['labels'].to(device)
                tgt_boxes  = tgt['boxes'].to(device)   # [num_gt, 4]
                q_idx, g_idx = self._match(pred_logits[i], pred_boxes[i],
                                           tgt_labels, tgt_boxes)

                # Classification: all queries default to no-object (= num_classes)
                tgt_cls = torch.full((pred_logits.shape[1],), self.num_classes,
                                     dtype=torch.long, device=device)
                if len(q_idx):
                    tgt_cls[q_idx] = tgt_labels[g_idx]

                weights = torch.ones(pred_logits.shape[1], device=device)
                weights[tgt_cls == self.num_classes] = self.no_object_weight
                loss_ce = F.cross_entropy(pred_logits[i],
                                          tgt_cls,
                                          reduction='none')
                total_ce = total_ce + (loss_ce * weights).mean()

                if len(q_idx):
                    matched_pred = pred_boxes[i][q_idx]   # [M, 4]
                    matched_gt   = tgt_boxes[g_idx]       # [M, 4]
                    total_bbox = total_bbox + F.l1_loss(matched_pred, matched_gt)
                    giou_mat = generalized_box_iou(matched_pred, matched_gt)
                    total_giou = total_giou + (1 - giou_mat.diagonal()).mean()
                    num_matched += len(q_idx)

            bs = len(targets)
            losses = {
                'loss_ce':    total_ce / bs,
                'loss_bbox':  total_bbox / max(num_matched, 1),
                'loss_giou':  total_giou / max(num_matched, 1),
                'loss_total': (self.weight_dict['loss_ce']   * total_ce / bs
                               + self.weight_dict['loss_bbox'] * total_bbox / max(num_matched, 1)
                               + self.weight_dict['loss_giou'] * total_giou / max(num_matched, 1)),
            }
            return losses

    return DETRLoss(num_classes)


def train_one_epoch(model, criterion, data_loader, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    criterion.train()
    
    loss_meter = AverageMeter('Loss')
    
    pbar = tqdm(data_loader, desc=f'Epoch {epoch}')
    
    for images, targets in pbar:
        images = images.to(device)
        
        # Forward
        outputs = model(images)
        losses = criterion(outputs, targets)
        loss = losses['loss_total']
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update meters
        loss_meter.update(loss.item(), images.size(0))
        
        # Update progress bar
        pbar.set_postfix({'loss': f'{loss_meter.avg:.4f}'})
    
    return {
        'loss': loss_meter.avg,
        'loss_ce': losses.get('loss_ce', 0).item() if torch.is_tensor(losses.get('loss_ce')) else 0,
        'loss_bbox': losses.get('loss_bbox', 0).item() if torch.is_tensor(losses.get('loss_bbox')) else 0,
    }


@torch.no_grad()
def evaluate(model, criterion, data_loader, device):
    """Evaluate model."""
    model.eval()
    criterion.eval()
    
    loss_meter = AverageMeter('Loss')
    
    for images, targets in tqdm(data_loader, desc='Validation'):
        images = images.to(device)
        
        outputs = model(images)
        losses = criterion(outputs, targets)
        loss = losses['loss_total']
        
        loss_meter.update(loss.item(), images.size(0))
    
    return {'val_loss': loss_meter.avg}


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train DETR model for steel defect detection')
    
    parser.add_argument('--data', type=str, default='data/processed/coco',
                       help='Path to COCO format data directory (legacy, not required)')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config file (optional)')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--img-size', type=int, default=640,
                       help='Image size')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--device', type=str, default=None,
                       help='Device (cuda:0, cpu, etc.)')
    parser.add_argument('--workers', type=int, default=4,
                       help='Number of workers for data loading')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--output-dir', type=str, default='results/models',
                       help='Output directory for models')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint')
    parser.add_argument('--splits', type=str, default='data/processed/splits.json',
                       help='Path to splits.json')
    parser.add_argument('--annotations', type=str,
                        default='data/raw/NEU-DET/ANNOTATIONS',
                        help='Path to VOC-XML annotations directory')
    parser.add_argument('--num-queries', type=int, default=30,
                       help='Number of object queries (smaller = faster on CPU)')
    parser.add_argument('--hidden-dim', type=int, default=128,
                       help='Transformer hidden dimension (smaller = faster on CPU)')
    
    return parser.parse_args()


def train_detr(args):
    """
    Train DETR model.
    
    Args:
        args: Command line arguments
    """
    # Setup
    set_seed(args.seed)
    setup_logging('experiments/logs', 'detr_training.log')
    
    logger.info("="*80)
    logger.info("DETR Training for Steel Surface Defect Detection")
    logger.info("="*80)
    
    # Save environment info
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    env_file = output_dir / 'env.json'
    env_info = save_environment_info(str(env_file))
    logger.info(f"Environment: {json.dumps(env_info, indent=2)}")
    
    # Device
    device = get_device(args.device)
    
    # Build model
    logger.info("Building DETR-Lite model...")
    num_queries = getattr(args, 'num_queries', 30)
    hidden_dim  = getattr(args, 'hidden_dim', 128)
    model = DETRLite(num_classes=6, num_queries=num_queries, hidden_dim=hidden_dim)
    model = model.to(device)
    logger.info(f"Queries: {num_queries}, hidden_dim: {hidden_dim}")
    
    n_parameters = count_parameters(model)
    logger.info(f"Model parameters: {n_parameters:,}")
    
    # Build criterion
    criterion = build_criterion(num_classes=6)
    criterion = criterion.to(device)
    
    # Build optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
    
    # Build datasets
    logger.info("Loading datasets...")
    splits_path = getattr(args, 'splits', 'data/processed/splits.json')
    annotations_dir = getattr(args, 'annotations', 'data/raw/NEU-DET/ANNOTATIONS')
    with open(splits_path) as _f:
        _splits = json.load(_f)
    train_dataset = SteelDETRDataset(_splits['train'], annotations_dir, args.img_size)
    val_dataset   = SteelDETRDataset(_splits['val'],   annotations_dir, args.img_size)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, collate_fn=collate_fn_detr
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, collate_fn=collate_fn_detr
    )
    
    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Training loop
    logger.info("Starting training...")
    start_time = time.time()
    
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'lr': []}
    
    for epoch in range(1, args.epochs + 1):
        logger.info(f"\nEpoch {epoch}/{args.epochs}")
        logger.info(f"Learning rate: {get_lr(optimizer):.6f}")
        
        # Train
        train_metrics = train_one_epoch(model, criterion, train_loader, optimizer, device, epoch)
        
        # Validate
        val_metrics = evaluate(model, criterion, val_loader, device)
        
        # Update scheduler
        lr_scheduler.step()
        
        # Log metrics
        logger.info(f"Train loss: {train_metrics['loss']:.4f}")
        logger.info(f"Val loss: {val_metrics['val_loss']:.4f}")
        
        # Save history
        history['train_loss'].append(train_metrics['loss'])
        history['val_loss'].append(val_metrics['val_loss'])
        history['lr'].append(get_lr(optimizer))
        
        # Save best model
        if val_metrics['val_loss'] < best_val_loss:
            best_val_loss = val_metrics['val_loss']
            best_model_path = output_dir / 'detr_best.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['val_loss'],
            }, best_model_path)
            logger.info(f"Best model saved to {best_model_path}")
        
        # Save checkpoint periodically
        if epoch % 10 == 0:
            ckpt_path = output_dir / f'detr_epoch_{epoch}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, ckpt_path)
    
    # Training complete
    total_time = time.time() - start_time
    logger.info("="*80)
    logger.info(f"Training completed in {format_time(total_time)}")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    
    # Save training metrics
    metrics = {
        'model': 'DETR-Lite',
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'img_size': args.img_size,
        'learning_rate': args.lr,
        'device': str(device),
        'seed': args.seed,
        'total_time_seconds': total_time,
        'best_val_loss': best_val_loss,
        'num_parameters': n_parameters,
        'history': history,
        'training_complete': True,
    }
    
    metrics_file = output_dir.parent / 'metrics' / 'detr_training.json'
    save_metrics(metrics, str(metrics_file))
    logger.info(f"Training metrics saved to {metrics_file}")
    logger.info("="*80)


def main():
    """Main entry point."""
    args = parse_args()
    train_detr(args)


if __name__ == '__main__':
    main()
