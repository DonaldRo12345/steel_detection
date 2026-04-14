"""
Evaluate the effect of CLAHE preprocessing at inference time.

Compares mAP of existing yolo_best.pt on:
  1. Original test images (baseline)
  2. CLAHE-processed test images (enhancement)

This isolates the CLAHE contribution without retraining.
"""
import sys, json, shutil, time
from pathlib import Path

sys.path.insert(0, 'src')

from ultralytics import YOLO

MODEL   = Path('results/models/yolo_best.pt')
YAML    = Path('data/processed/yolo/data.yaml')
CLAHE_YAML = Path('data/processed/yolo_clahe/data.yaml')

print("=" * 60)
print("CLAHE Inference-Time Evaluation")
print("=" * 60)

m = YOLO(str(MODEL))

# ── 1. Baseline eval (original images) ────────────────────────────────────
print("\n[1/2] Evaluating on ORIGINAL test images ...")
t0 = time.time()
res1 = m.val(data=str(YAML), split='test', verbose=False, workers=0)
t1 = time.time() - t0
map50_base  = float(res1.box.map50)
map5095_base = float(res1.box.map)
print(f"  mAP@0.5     = {map50_base:.4f}")
print(f"  mAP@0.5:0.95= {map5095_base:.4f}")
print(f"  Time: {t1:.0f}s")

# ── 2. CLAHE eval ──────────────────────────────────────────────────────────
if CLAHE_YAML.exists():
    print("\n[2/2] Evaluating on CLAHE-processed test images ...")
    t0 = time.time()
    res2 = m.val(data=str(CLAHE_YAML), split='test', verbose=False, workers=0)
    t2 = time.time() - t0
    map50_clahe  = float(res2.box.map50)
    map5095_clahe = float(res2.box.map)
    print(f"  mAP@0.5     = {map50_clahe:.4f}")
    print(f"  mAP@0.5:0.95= {map5095_clahe:.4f}")
    print(f"  Time: {t2:.0f}s")
    delta = map50_clahe - map50_base
    print(f"\nCLAHE delta mAP@0.5 = {delta:+.4f} "
          f"({'improvement' if delta > 0 else 'regression'})")

    result = {
        'baseline_mAP50': map50_base,
        'baseline_mAP50_95': map5095_base,
        'clahe_inference_mAP50': map50_clahe,
        'clahe_inference_mAP50_95': map5095_clahe,
        'delta_mAP50': delta,
    }
    with open('results/metrics/clahe_inference_eval.json', 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to results/metrics/clahe_inference_eval.json")
else:
    print(f"CLAHE dataset not found at {CLAHE_YAML}")
    print("Run with --clahe flag in train_yolo_enhanced.py first")

print("\nDone.")
