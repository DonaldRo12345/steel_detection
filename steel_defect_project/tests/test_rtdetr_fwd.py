"""Minimal RT-DETR val test — bypass hanging by using a write-and-run script."""
from ultralytics import RTDETR
import time

DATA = 'data/processed/yolo/data.yaml'
MODEL = 'rtdetr-l.pt'

print("Loading RT-DETR ...")
m = RTDETR(MODEL)
print("Loaded.")

# Quick 1-batch forward pass test
import torch
import numpy as np

# Create a random batch
imgs = torch.rand(1, 3, 640, 640)
print("Forward pass test ...")
t0 = time.time()
with torch.no_grad():
    out = m.model(imgs)
print(f"Forward pass: {time.time()-t0:.2f}s")
print("Output type:", type(out))
if hasattr(out, '__len__'):
    print("Output len:", len(out))
print("RT-DETR forward OK")
