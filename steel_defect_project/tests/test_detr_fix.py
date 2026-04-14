"""Quick smoke-test for the fixed DETR dataset, model, and loss."""
import sys, json, torch
sys.path.insert(0, 'src')

from train_detr import (DETRLite, SteelDETRDataset, collate_fn_detr,
                        build_criterion, CLASS_NAMES_DETR)
from torch.utils.data import DataLoader

with open('data/processed/splits.json') as f:
    splits = json.load(f)

# Dataset
ds = SteelDETRDataset(splits['train'][:8], 'data/raw/NEU-DET/ANNOTATIONS', img_size=320)
print(f'Dataset length: {len(ds)}')
img, tgt = ds[0]
print(f'Image shape: {img.shape}')
print(f'Target labels: {tgt["labels"]}')
print(f'Target boxes:  {tgt["boxes"].shape}')

# DataLoader
loader = DataLoader(ds, batch_size=4, collate_fn=collate_fn_detr)
imgs, tgts = next(iter(loader))
print(f'Batch images: {imgs.shape}')
print(f'Batch targets: {len(tgts)} items, first has {len(tgts[0]["labels"])} boxes')

# Model forward (train mode so gradients flow)
model = DETRLite(num_classes=6, num_queries=20, hidden_dim=128)
model.train()
out = model(imgs)
print(f'pred_logits: {out["pred_logits"].shape}')  # [B, 20, 7]
print(f'pred_boxes:  {out["pred_boxes"].shape}')   # [B, 20, 4]

# Loss (Hungarian matching)
crit = build_criterion(6)
losses = crit(out, tgts)
print('Losses:', {k: round(float(v), 4) for k, v in losses.items()})

# Backward
total = losses['loss_total']
total.backward()
print(f'Backward OK, loss_total = {total.item():.4f}')

# Eval mode (inference)
model.eval()
with torch.no_grad():
    out2 = model(imgs)
print(f'Eval forward OK: pred_logits {out2["pred_logits"].shape}')
print('ALL OK')
