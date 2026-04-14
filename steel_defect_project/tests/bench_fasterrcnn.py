"""Benchmark Faster R-CNN timing: 1 epoch with frozen backbone."""
import sys, json, time
sys.path.insert(0, 'src')

import torch
from torch.utils.data import DataLoader
from train_fasterrcnn import (SteelDefectDataset, build_fasterrcnn,
                              collate_fn, CLASS_NAMES)

with open('data/processed/splits.json') as f:
    splits = json.load(f)

# Small subset to estimate timing
sample = splits['train'][:80]  # 80 images, ~20 batches
ds = SteelDefectDataset(sample, 'data/raw/NEU-DET/ANNOTATIONS', img_size=320)
loader = DataLoader(ds, batch_size=4, shuffle=False,
                    num_workers=0, collate_fn=collate_fn)

num_classes = len(CLASS_NAMES) + 1
model = build_fasterrcnn(num_classes, pretrained=True, freeze_backbone=True)
model.train()
device = torch.device('cpu')

import torch.optim as optim
params = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

t0 = time.time()
for i, (images, targets) in enumerate(loader):
    images = [img.to(device) for img in images]
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
    loss_dict = model(images, targets)
    losses = sum(loss for loss in loss_dict.values())
    optimizer.zero_grad()
    losses.backward()
    optimizer.step()
    elapsed = time.time() - t0
    if i == 2:
        sec_per_batch = elapsed / (i + 1)
        total_batches_per_epoch = len(splits['train']) / 4
        est_epoch_min = sec_per_batch * total_batches_per_epoch / 60
        print(f"Avg sec/batch (frozen backbone, 320px): {sec_per_batch:.1f}s")
        print(f"Estimated minutes per full epoch: {est_epoch_min:.0f} min")
        print(f"Estimated 10 epochs: {est_epoch_min*10/60:.1f} hours")
        break

print("Timing benchmark done")
