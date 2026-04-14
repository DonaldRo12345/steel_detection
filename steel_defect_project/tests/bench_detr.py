"""Benchmark DETR-Lite timing on real data."""
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
ds = SteelDETRDataset(sample, 'data/raw/NEU-DET/ANNOTATIONS', img_size=640)
loader = DataLoader(ds, batch_size=4, shuffle=False,
                    num_workers=0, collate_fn=collate_fn_detr)

model = DETRLite(num_classes=6, num_queries=100, hidden_dim=256)
model.train()
criterion = build_criterion(6)
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
device = torch.device('cpu')

t0 = time.time()
for i, (imgs, tgts) in enumerate(loader):
    imgs = imgs.to(device)
    out = model(imgs)
    losses = criterion(out, tgts)
    loss = losses['loss_total']
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if i == 3:
        elapsed = time.time() - t0
        spb = elapsed / (i + 1)
        n_batches = len(splits['train']) / 4
        est_min = spb * n_batches / 60
        print(f"Avg sec/batch (DETR-Lite, 640px, batch 4): {spb:.1f}s")
        print(f"Est minutes/epoch: {est_min:.0f}")
        print(f"Est 15 epochs: {est_min*15/60:.1f} hours")
        break

print("DETR bench done")
