"""Benchmark DETR-Lite at 320px image size."""
import sys, json, time
sys.path.insert(0, 'src')

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from train_detr import (SteelDETRDataset, collate_fn_detr, DETRLite,
                        build_criterion)

with open('data/processed/splits.json') as f:
    splits = json.load(f)

sample = splits['train'][:32]
ds = SteelDETRDataset(sample, 'data/raw/NEU-DET/ANNOTATIONS', img_size=320)
loader = DataLoader(ds, batch_size=8, shuffle=False,
                    num_workers=0, collate_fn=collate_fn_detr)

model = DETRLite(num_classes=6, num_queries=100, hidden_dim=256)
model.train()
criterion = build_criterion(6)
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

t0 = time.time()
for i, (imgs, tgts) in enumerate(loader):
    out = model(imgs)
    losses = criterion(out, tgts)
    losses['loss_total'].backward()
    optimizer.step()
    optimizer.zero_grad()
    if i == 2:
        spb = (time.time() - t0) / (i + 1)
        n_batches = len(splits['train']) / 8
        est_min = spb * n_batches / 60
        print(f"320px, batch 8 — sec/batch: {spb:.1f}s")
        print(f"Est min/epoch: {est_min:.0f}")
        print(f"Est 10 epochs: {est_min*10/60:.1f} hours")
        break
print("Done")
