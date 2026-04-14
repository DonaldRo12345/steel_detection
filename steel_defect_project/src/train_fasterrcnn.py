"""
Faster R-CNN training script for steel surface defect detection.
Uses torchvision's pretrained Faster R-CNN with ResNet-50 FPN backbone.
Serves as a classic two-stage detector baseline.
"""

import os
import json
import argparse
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms.v2 as T
from tqdm import tqdm
import xml.etree.ElementTree as ET

from utils import (set_seed, get_device, save_environment_info, save_metrics,
                   setup_logging, AverageMeter, format_time, count_parameters, get_lr)


logger = logging.getLogger(__name__)


# ── Dataset ────────────────────────────────────────────────────────────────────

CLASS_NAMES = ['Crazing', 'Inclusion', 'Patches', 'Pitted_surface',
               'Rolled-in_scale', 'Scratches']
CLASS_NAME_MAPPING = {
    'crazing': 'Crazing', 'inclusion': 'Inclusion', 'patches': 'Patches',
    'pitted_surface': 'Pitted_surface', 'pitted surface': 'Pitted_surface',
    'rolled-in_scale': 'Rolled-in_scale', 'rolled-in scale': 'Rolled-in_scale',
    'scratches': 'Scratches',
}
CLASS_TO_IDX = {c: i + 1 for i, c in enumerate(CLASS_NAMES)}  # 0 = background


class SteelDefectDataset(Dataset):
    """Dataset loader for NEU-DET / generic steel defect datasets.

    Reads image paths from a split JSON, parses matching XML annotations,
    and returns dicts compatible with torchvision detection models.
    """

    def __init__(self, image_paths: List[str], annotations_dir: str,
                 img_size: int = 640, transforms=None):
        self.image_paths = image_paths
        self.annotations_dir = Path(annotations_dir)
        self.img_size = img_size
        self.transforms = transforms

    def __len__(self):
        return len(self.image_paths)

    def _parse_xml(self, xml_path: str):
        boxes, labels = [], []
        if not os.path.exists(xml_path):
            return boxes, labels
        tree = ET.parse(xml_path)
        for obj in tree.getroot().findall('object'):
            name = CLASS_NAME_MAPPING.get(obj.find('name').text.lower(),
                                           obj.find('name').text)
            if name not in CLASS_TO_IDX:
                continue
            bb = obj.find('bndbox')
            boxes.append([int(bb.find(t).text) for t in ('xmin', 'ymin', 'xmax', 'ymax')])
            labels.append(CLASS_TO_IDX[name])
        return boxes, labels

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = cv2.imread(img_path)
        if img is None:
            # Return a small dummy sample on read failure
            img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))

        orig_h, orig_w = cv2.imread(img_path).shape[:2] if os.path.exists(img_path) else (self.img_size, self.img_size)
        scale_x = self.img_size / orig_w
        scale_y = self.img_size / orig_h

        xml_path = self.annotations_dir / (Path(img_path).stem + '.xml')
        boxes, labels = self._parse_xml(str(xml_path))

        if boxes:
            boxes_t = torch.as_tensor(boxes, dtype=torch.float32)
            boxes_t[:, [0, 2]] *= scale_x
            boxes_t[:, [1, 3]] *= scale_y
        else:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)

        labels_t = torch.as_tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64)
        area = (boxes_t[:, 2] - boxes_t[:, 0]) * (boxes_t[:, 3] - boxes_t[:, 1]) if len(boxes_t) else torch.zeros((0,))

        target = {
            'boxes': boxes_t,
            'labels': labels_t,
            'image_id': torch.tensor([idx]),
            'area': area,
            'iscrowd': torch.zeros((len(labels_t),), dtype=torch.int64),
        }

        img_tensor = torch.as_tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0

        if self.transforms:
            img_tensor = self.transforms(img_tensor)

        return img_tensor, target


def collate_fn(batch):
    return tuple(zip(*batch))


# ── Model ──────────────────────────────────────────────────────────────────────

def build_fasterrcnn(num_classes: int = 7, pretrained: bool = True,
                     freeze_backbone: bool = False):
    """Build Faster R-CNN with ResNet-50-FPN backbone.

    Args:
        num_classes: number of classes **including background** (6 defects + 1 bg = 7).
        pretrained: use COCO-pretrained weights.
        freeze_backbone: if True, freeze backbone + FPN so only the RPN and RoI
            head are trained, which is much faster on CPU.
    """
    if pretrained:
        model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    else:
        model = fasterrcnn_resnet50_fpn(weights=None)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    if freeze_backbone:
        # Freeze backbone body and FPN; only RPN + RoI head remain trainable
        for name, param in model.named_parameters():
            if 'roi_heads.box_predictor' not in name and 'rpn' not in name:
                param.requires_grad = False
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Backbone frozen — trainable params: {trainable:,}")

    return model


# ── Training ───────────────────────────────────────────────────────────────────

def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    loss_meter = AverageMeter('Loss')

    pbar = tqdm(data_loader, desc=f'Epoch {epoch}')
    for images, targets in pbar:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        loss_meter.update(losses.item(), len(images))
        pbar.set_postfix({'loss': f'{loss_meter.avg:.4f}'})

    return {'loss': loss_meter.avg}


@torch.no_grad()
def evaluate_epoch(model, data_loader, device):
    model.train()  # Faster R-CNN returns losses only in train mode
    loss_meter = AverageMeter('Loss')

    for images, targets in tqdm(data_loader, desc='Validation'):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_meter.update(losses.item(), len(images))

    return {'val_loss': loss_meter.avg}


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='Train Faster R-CNN for steel defect detection')
    p.add_argument('--data', required=True, help='Path to splits.json')
    p.add_argument('--annotations', required=True, help='Path to annotations directory')
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--img-size', type=int, default=640)
    p.add_argument('--lr', type=float, default=0.005)
    p.add_argument('--momentum', type=float, default=0.9)
    p.add_argument('--weight-decay', type=float, default=0.0005)
    p.add_argument('--device', type=str, default=None)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output-dir', type=str, default='results/models')
    p.add_argument('--pretrained', action='store_true', default=True)
    p.add_argument('--resume', type=str, default=None)
    p.add_argument('--freeze-backbone', action='store_true', default=False,
                   help='Freeze backbone+FPN; only train RPN+RoI head (fast on CPU)')
    return p.parse_args()


def train_fasterrcnn(args):
    set_seed(args.seed)
    setup_logging('experiments/logs', 'fasterrcnn_training.log')

    logger.info("=" * 80)
    logger.info("Faster R-CNN Training for Steel Surface Defect Detection")
    logger.info("=" * 80)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    env_file = output_dir / 'env.json'
    save_environment_info(str(env_file))

    device = get_device(args.device)

    # Build model
    num_classes = len(CLASS_NAMES) + 1  # +1 for background
    model = build_fasterrcnn(num_classes=num_classes, pretrained=args.pretrained,
                              freeze_backbone=getattr(args, 'freeze_backbone', False))
    model.to(device)
    logger.info(f"Faster R-CNN parameters: {count_parameters(model):,}")

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        logger.info(f"Resumed from {args.resume}")

    # Optimizer & scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    # Datasets
    with open(args.data) as f:
        splits = json.load(f)

    train_ds = SteelDefectDataset(splits['train'], args.annotations, args.img_size)
    val_ds = SteelDefectDataset(splits['val'], args.annotations, args.img_size)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, collate_fn=collate_fn, pin_memory=True)

    logger.info(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    # Training loop
    start_time = time.time()
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'lr': []}

    for epoch in range(1, args.epochs + 1):
        logger.info(f"\nEpoch {epoch}/{args.epochs}  lr={get_lr(optimizer):.6f}")

        train_metrics = train_one_epoch(model, optimizer, train_loader, device, epoch)
        val_metrics = evaluate_epoch(model, val_loader, device)
        lr_scheduler.step()

        logger.info(f"Train loss: {train_metrics['loss']:.4f}  Val loss: {val_metrics['val_loss']:.4f}")

        history['train_loss'].append(train_metrics['loss'])
        history['val_loss'].append(val_metrics['val_loss'])
        history['lr'].append(get_lr(optimizer))

        if val_metrics['val_loss'] < best_val_loss:
            best_val_loss = val_metrics['val_loss']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
            }, output_dir / 'fasterrcnn_best.pth')
            logger.info("  ↳ saved best model")

        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, output_dir / f'fasterrcnn_epoch_{epoch}.pth')

    total_time = time.time() - start_time
    logger.info("=" * 80)
    logger.info(f"Training finished in {format_time(total_time)}")
    logger.info(f"Best val loss: {best_val_loss:.4f}")

    metrics = {
        'model': 'Faster R-CNN (ResNet50-FPN)',
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'img_size': args.img_size,
        'learning_rate': args.lr,
        'device': str(device),
        'seed': args.seed,
        'total_time_seconds': total_time,
        'best_val_loss': best_val_loss,
        'num_parameters': count_parameters(model),
        'history': history,
        'training_complete': True,
    }
    metrics_file = output_dir.parent / 'metrics' / 'fasterrcnn_training.json'
    save_metrics(metrics, str(metrics_file))
    logger.info(f"Metrics saved to {metrics_file}")


def main():
    args = parse_args()
    train_fasterrcnn(args)


if __name__ == '__main__':
    main()
