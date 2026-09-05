"""Evaluate Enhanced YOLOv8 on test set and save per-class metrics."""
import sys
sys.path.insert(0, 'src')
import json
from ultralytics import YOLO

MODEL = 'results/models/yolo_enhanced_30ep/weights/best.pt'
YAML  = 'data/processed/yolo_clahe/data.yaml'
OUT   = 'results/metrics/eval_yolo_enhanced.json'
CLASS_NAMES = ['Crazing', 'Inclusion', 'Patches', 'Pitted_surface',
               'Rolled-in_scale', 'Scratches']

m = YOLO(MODEL)
res = m.val(data=YAML, split='test', verbose=False, workers=0)

per_class = {}
if hasattr(res.box, 'ap_class_index') and res.box.ap_class_index is not None:
    for i, cls_idx in enumerate(res.box.ap_class_index):
        name = CLASS_NAMES[int(cls_idx)]
        per_class[str(int(cls_idx))] = {
            'class': name,
            'AP50': float(res.box.ap50[i]),
        }

result = {
    'model': MODEL,
    'data': YAML,
    'mAP_50': float(res.box.map50),
    'mAP_50_95': float(res.box.map),
    'precision': float(res.box.mp),
    'recall': float(res.box.mr),
    'per_class_AP': {k: v['AP50'] for k, v in per_class.items()},
}

print(f"mAP@0.5:      {result['mAP_50']:.4f}")
print(f"mAP@0.5:0.95: {result['mAP_50_95']:.4f}")
print(f"Precision:    {result['precision']:.4f}")
print(f"Recall:       {result['recall']:.4f}")
print()
print("Per-class AP@0.5:")
for cid, v in per_class.items():
    print(f"  {v['class']:<20}: {v['AP50']:.4f}")

with open(OUT, 'w') as f:
    json.dump(result, f, indent=2)
print(f"\nSaved to {OUT}")
