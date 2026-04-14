"""Extract per-class metrics from YOLO val results."""
import sys
sys.path.insert(0, 'src')

import json
from ultralytics import YOLO
from pathlib import Path

MODEL = Path('results/models/yolo_best.pt')
YAML  = Path('data/processed/yolo/data.yaml')
CLASS_NAMES = ['Crazing', 'Inclusion', 'Patches', 'Pitted_surface',
               'Rolled-in_scale', 'Scratches']

m = YOLO(str(MODEL))
res = m.val(data=str(YAML), split='test', verbose=False, workers=0)

per_class = {}
if hasattr(res.box, 'ap_class_index') and res.box.ap_class_index is not None:
    for i, cls_idx in enumerate(res.box.ap_class_index):
        name = CLASS_NAMES[int(cls_idx)]
        per_class[str(int(cls_idx))] = {
            'class': name,
            'AP50': float(res.box.ap50[i]),
            'AP50_95': float(res.box.ap[i]),
            'P': float(res.box.p[i]) if hasattr(res.box, 'p') and res.box.p is not None else None,
            'R': float(res.box.r[i]) if hasattr(res.box, 'r') and res.box.r is not None else None,
        }

result = {
    'model_type': 'YOLOv8n',
    'model_path': str(MODEL),
    'mAP_50': float(res.box.map50),
    'mAP_50_95': float(res.box.map),
    'precision': float(res.box.mp),
    'recall': float(res.box.mr),
    'per_class_AP': {k: v['AP50'] for k, v in per_class.items()},
    'per_class_details': per_class,
}

print(f"Overall mAP@0.5:     {result['mAP_50']:.4f}")
print(f"Overall mAP@0.5:0.95:{result['mAP_50_95']:.4f}")
print(f"Precision: {result['precision']:.4f}")
print(f"Recall:    {result['recall']:.4f}")
print()
print("Per-class AP@0.5:")
for cid, v in per_class.items():
    print(f"  [{cid}] {v['class']:<20}: AP50={v['AP50']:.4f}")

with open('results/metrics/eval_yolo_perclass.json', 'w') as f:
    json.dump(result, f, indent=2)
print('\nSaved to results/metrics/eval_yolo_perclass.json')
