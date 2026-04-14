"""Quick timing benchmark for RT-DETR training."""
import sys, time
sys.path.insert(0, 'src')

from ultralytics import RTDETR
from pathlib import Path

DATA = str(Path('data/processed/yolo/data.yaml').resolve())
MODEL = 'rtdetr-l.pt'

print("Benchmarking RT-DETR-L training speed (3 batches) ...")
m = RTDETR(MODEL)

# Override progress so we can measure time
import torch; torch.manual_seed(0)

t0 = time.time()
# Train for 1 epoch on just 5% of data to estimate time
results = m.train(
    data=DATA,
    epochs=1,
    batch=4,
    imgsz=640,
    device='cpu',
    workers=0,
    fraction=0.05,    # use only 5% of training data
    verbose=False,
    plots=False,
    save=False,
)
elapsed = time.time() - t0

# Extrapolate to full dataset: 5% → 100% means multiply by 20
# And 5 epochs = multiply by 5
est_full_epoch_min = elapsed * 20 / 60
print(f"\n5% data, 1 epoch: {elapsed:.0f}s")
print(f"Estimated full epoch: {est_full_epoch_min:.0f} min")
print(f"Estimated 5 epochs:   {est_full_epoch_min*5/60:.1f} hours")
print(f"Estimated 10 epochs:  {est_full_epoch_min*10/60:.1f} hours")
