"""
Evaluate DETR-Lite model on NEU-DET test split.
Usage:
  cd steel_defect_project
  python tests/eval_detr.py --model results/models/detr_best.pth \
         --splits data/processed/splits.json \
         --annotations data/raw/NEU-DET/ANNOTATIONS \
         --out results/metrics/eval_detr_real.json
"""
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
import cv2
from tqdm import tqdm

# Allow imports from src/
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from train_detr import DETRLite, SteelDETRDataset, CLASS_NAMES_DETR, collate_fn_detr
from torch.utils.data import DataLoader


# ── IoU helpers ───────────────────────────────────────────────────────────────
def cxcywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """Convert (cx,cy,w,h) → (x1,y1,x2,y2).  Input shape: (N,4)."""
    out = np.zeros_like(boxes)
    out[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    out[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    out[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    out[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return out


def iou_single(b1, b2):
    """b1, b2: (4,) in xyxy."""
    xi1, yi1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    xi2, yi2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


# ── AP computation ─────────────────────────────────────────────────────────────
def compute_ap(rec, prec):
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


# ── Main evaluation ───────────────────────────────────────────────────────────
def evaluate(args):
    device = torch.device("cpu")

    # Load splits.json
    with open(args.splits) as f:
        splits = json.load(f)
    test_paths = splits.get("test", [])
    print(f"Test images: {len(test_paths)}")

    # Dataset
    dataset = SteelDETRDataset(test_paths, args.annotations, img_size=320)
    loader = DataLoader(dataset, batch_size=8, shuffle=False,
                        collate_fn=collate_fn_detr, num_workers=0)

    # Load model
    checkpoint = torch.load(args.model, map_location="cpu", weights_only=False)
    # Extract args/config stored in checkpoint
    num_queries = checkpoint.get("num_queries", 30)
    hidden_dim  = checkpoint.get("hidden_dim", 128)

    model = DETRLite(num_classes=6, num_queries=num_queries, hidden_dim=hidden_dim,
                     nheads=8, num_encoder_layers=3, num_decoder_layers=3)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Loaded model from epoch {checkpoint.get('epoch','?')}, "
          f"num_queries={num_queries}, hidden_dim={hidden_dim}")

    # ── Collect all predictions & ground-truths per class ──────────────────
    num_classes = 6
    # per_class_preds[c] = list of (score, tp) sorted by score desc
    per_class_preds = [[] for _ in range(num_classes)]
    per_class_ngt   = [0] * num_classes  # total GT boxes per class

    conf_thresh = args.conf_thresh
    iou_thresh  = 0.5

    total_images = 0
    total_latency = 0.0

    with torch.no_grad():
        for images, targets in tqdm(loader, desc="Evaluating"):
            t0 = time.time()
            outputs = model(images)
            total_latency += time.time() - t0
            total_images += images.shape[0]

            pred_logits = outputs["pred_logits"]   # (B, Q, C+1)
            pred_boxes  = outputs["pred_boxes"]    # (B, Q, 4) cxcywh [0,1]

            for bi in range(images.shape[0]):
                logits = pred_logits[bi]   # (Q, C+1)
                boxes  = pred_boxes[bi]    # (Q, 4)
                target = targets[bi]       # dict with 'labels', 'boxes'

                # Softmax → class probabilities (drop no-object class index=6)
                probs = torch.softmax(logits, dim=-1)[:, :num_classes]  # (Q, C)
                scores, pred_cls = probs.max(dim=-1)  # (Q,)

                # Filter by confidence
                mask = scores >= conf_thresh
                scores  = scores[mask].cpu().numpy()
                pred_cls = pred_cls[mask].cpu().numpy()
                pred_bx  = cxcywh_to_xyxy(boxes[mask].cpu().numpy())

                # Ground truth
                gt_labels = target["labels"].cpu().numpy()   # (M,)
                gt_boxes  = cxcywh_to_xyxy(target["boxes"].cpu().numpy())  # (M,4)

                # Count GT per class
                for c in gt_labels:
                    per_class_ngt[int(c)] += 1

                # Match predictions to GT per class
                gt_used = [False] * len(gt_labels)
                # Sort preds by score desc
                order = np.argsort(-scores)
                for idx in order:
                    cls = int(pred_cls[idx])
                    sc  = float(scores[idx])
                    bx  = pred_bx[idx]
                    # Find best-matching GT of same class
                    best_iou, best_j = 0.0, -1
                    for j, (gc, gb) in enumerate(zip(gt_labels, gt_boxes)):
                        if gc != cls or gt_used[j]:
                            continue
                        iou = iou_single(bx, gb)
                        if iou > best_iou:
                            best_iou, best_j = iou, j
                    tp = 0
                    if best_iou >= iou_thresh:
                        tp = 1
                        gt_used[best_j] = True
                    per_class_preds[cls].append((sc, tp))

    # ── Compute per-class AP ───────────────────────────────────────────────
    aps = []
    print("\n=== Per-Class Results ===")
    print(f"{'Class':<20} {'AP@0.5':>8}  {'#GT':>6}  {'#Pred':>6}")
    for c, name in enumerate(CLASS_NAMES_DETR):
        ngt = per_class_ngt[c]
        preds = sorted(per_class_preds[c], key=lambda x: -x[0])
        if not preds or ngt == 0:
            print(f"  {name:<18} {'N/A':>8}  {ngt:>6}  {len(preds):>6}")
            continue
        tps = np.array([p[1] for p in preds])
        fps = 1 - tps
        tp_cum = np.cumsum(tps)
        fp_cum = np.cumsum(fps)
        rec  = tp_cum / ngt
        prec = tp_cum / (tp_cum + fp_cum)
        ap   = compute_ap(rec, prec)
        aps.append(ap)
        print(f"  {name:<18} {ap:>8.4f}  {ngt:>6}  {len(preds):>6}")

    mean_ap = float(np.mean(aps)) if aps else 0.0
    avg_latency_ms = 1000 * total_latency / max(total_images, 1)
    print(f"\n  mAP@0.5     = {mean_ap:.4f}")
    print(f"  Avg latency = {avg_latency_ms:.1f} ms/image")

    results = {
        "model": args.model,
        "num_test_images": total_images,
        "mAP_50": round(mean_ap, 4),
        "per_class_AP50": {CLASS_NAMES_DETR[c]: round(float(v), 4)
                           for c, v in enumerate(aps) if c < len(aps)},
        "avg_latency_ms": round(avg_latency_ms, 1),
        "conf_thresh": conf_thresh,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to detr_best.pth")
    parser.add_argument("--splits", default="data/processed/splits.json")
    parser.add_argument("--annotations", default="data/raw/NEU-DET/ANNOTATIONS")
    parser.add_argument("--out", default="results/metrics/eval_detr_real.json")
    parser.add_argument("--conf-thresh", type=float, default=0.25)
    args = parser.parse_args()
    evaluate(args)
