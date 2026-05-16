"""
Steel Defect Detection — Gradio Web Interface (Hugging Face Spaces)
Supports: YOLOv8n baseline, Enhanced YOLOv8, DETR-Lite

Models are stored in the models/ subfolder of this Space.
src/train_detr.py must also be present for DETRLite class import.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from pathlib import Path
import numpy as np
import cv2
import torch
import torchvision.transforms as T
import gradio as gr
from ultralytics import YOLO
from PIL import Image

# ── Paths ──────────────────────────────────────────────────────────────────
# On HF Spaces __file__ is /home/user/app/app.py, so BASE = /home/user/app
BASE = Path(__file__).parent

YOLO_MODELS = {
    "YOLOv8n (Baseline) — mAP@0.5: 0.741":       BASE / "models/yolo_best.pt",
    "Enhanced YOLOv8 (CLAHE) — mAP@0.5: 0.664":  BASE / "models/yolo_enhanced_best.pt",
}

DETR_MODEL_KEY  = "DETR-Lite — mAP@0.5: 0.101"
DETR_MODEL_PATH = BASE / "models/detr_best.pth"

ALL_MODEL_KEYS = list(YOLO_MODELS.keys()) + [DETR_MODEL_KEY]

CLASS_NAMES = ["Crazing", "Inclusion", "Patches", "Pitted Surface",
               "Rolled-in Scale", "Scratches"]

CLASS_COLORS = [
    (220,  50,  50),  # Crazing      — red
    ( 50, 150, 220),  # Inclusion    — blue
    ( 50, 200,  80),  # Patches      — green
    (230, 140,  30),  # Pitted       — orange
    (160,  60, 220),  # Rolled-scale — purple
    ( 20, 200, 200),  # Scratches    — cyan
]

# ── Model cache ──────────────────────────────────────────────────────────────
_yolo_cache: dict = {}
_detr_model = None


def load_yolo(model_name: str):
    if model_name not in _yolo_cache:
        path = YOLO_MODELS[model_name]
        _yolo_cache[model_name] = YOLO(str(path))
    return _yolo_cache[model_name]


def load_detr():
    global _detr_model
    if _detr_model is None:
        from train_detr import DETRLite
        model = DETRLite(num_classes=6)
        ckpt = torch.load(str(DETR_MODEL_PATH), map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        _detr_model = model
    return _detr_model


# ── Preprocessing ────────────────────────────────────────────────────────────
def apply_clahe(img_np: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    if img_np.ndim == 2:
        return clahe.apply(img_np)
    channels = cv2.split(img_np)
    return cv2.merge([clahe.apply(c) for c in channels])


# ── Inference ────────────────────────────────────────────────────────────────
def _draw(annotated, x1, y1, x2, y2, label, score, color):
    cv2.rectangle(annotated, (x1, y1), (x2, y2), color[::-1], 2)
    text = f"{label} {score:.2f}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(annotated, (x1, y1 - th - 6), (x1 + tw + 4, y1), color[::-1], -1)
    cv2.putText(annotated, text, (x1 + 2, y1 - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


def run_yolo(model_name: str, pil_img: Image.Image, conf: float, iou: float):
    model = load_yolo(model_name)
    img_np = np.array(pil_img)
    if "CLAHE" in model_name:
        img_np = apply_clahe(img_np)

    results = model.predict(source=img_np, conf=conf, iou=iou, verbose=False, device="cpu")[0]
    annotated = img_np.copy()
    detections = []

    if results.boxes is not None and len(results.boxes):
        for box in results.boxes:
            cls_id = int(box.cls[0])
            score  = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            label = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id)
            color = CLASS_COLORS[cls_id % len(CLASS_COLORS)]
            _draw(annotated, x1, y1, x2, y2, label, score, color)
            detections.append({"Class": label, "Confidence": f"{score:.3f}",
                                "x1": x1, "y1": y1, "x2": x2, "y2": y2})

    return Image.fromarray(annotated), detections


_DETR_TRANSFORM = T.Compose([T.ToTensor(), T.Resize((640, 640))])


def run_detr(pil_img: Image.Image, conf: float):
    model = load_detr()
    img_np = np.array(pil_img)
    h, w = img_np.shape[:2]
    img_tensor = _DETR_TRANSFORM(img_np).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)

    logits = outputs["pred_logits"][0]
    boxes  = outputs["pred_boxes"][0]
    probs  = logits.softmax(-1)
    scores, labels = probs[:, :-1].max(-1)

    keep   = scores > conf
    scores = scores[keep].cpu().numpy()
    labels = labels[keep].cpu().numpy()
    boxes  = boxes[keep].cpu().numpy()

    annotated = img_np.copy()
    detections = []

    for score, cls_id, box in zip(scores, labels, boxes):
        cx, cy, bw, bh = box
        x1 = max(0, int((cx - bw / 2) * w))
        y1 = max(0, int((cy - bh / 2) * h))
        x2 = min(w, int((cx + bw / 2) * w))
        y2 = min(h, int((cy + bh / 2) * h))
        label = CLASS_NAMES[int(cls_id)] if int(cls_id) < len(CLASS_NAMES) else str(cls_id)
        color = CLASS_COLORS[int(cls_id) % len(CLASS_COLORS)]
        _draw(annotated, x1, y1, x2, y2, label, float(score), color)
        detections.append({"Class": label, "Confidence": f"{float(score):.3f}",
                            "x1": x1, "y1": y1, "x2": x2, "y2": y2})

    return Image.fromarray(annotated), detections


# ── Main detect function ──────────────────────────────────────────────────────
def detect(model_name, image, conf_thresh, iou_thresh):
    if image is None:
        return None, "No image uploaded.", []

    pil_img = Image.fromarray(image).convert("RGB")

    if model_name == DETR_MODEL_KEY:
        annotated, dets = run_detr(pil_img, conf_thresh)
    else:
        annotated, dets = run_yolo(model_name, pil_img, conf_thresh, iou_thresh)

    if not dets:
        summary = "No defects detected."
    else:
        counts = {}
        for d in dets:
            counts[d["Class"]] = counts.get(d["Class"], 0) + 1
        lines = [f"**{len(dets)} detections total**"]
        for cls, cnt in sorted(counts.items()):
            lines.append(f"- {cls}: {cnt}")
        summary = "\n".join(lines)

    table = [[d["Class"], d["Confidence"], d["x1"], d["y1"], d["x2"], d["y2"]]
             for d in dets]
    return np.array(annotated), summary, table


# ── UI ────────────────────────────────────────────────────────────────────────
with gr.Blocks(title="Steel Defect Detection") as demo:

    gr.Markdown("""
    # Steel Surface Defect Detection
    Upload a steel surface image and choose a model to detect defects.
    Supports 6 defect classes: **Crazing, Inclusion, Patches, Pitted Surface, Rolled-in Scale, Scratches**.

    | Model | mAP@0.5 | Notes |
    |---|---|---|
    | YOLOv8n Baseline | 0.741 | Standard training |
    | Enhanced YOLOv8 (CLAHE) | 0.664 | CLAHE contrast enhancement |
    | DETR-Lite | 0.101 | Transformer-based, 15 epochs |
    """)

    with gr.Row():
        with gr.Column(scale=1):
            model_dd  = gr.Dropdown(choices=ALL_MODEL_KEYS, value=ALL_MODEL_KEYS[0], label="Model")
            image_in  = gr.Image(label="Upload Image", type="numpy")
            conf_sl   = gr.Slider(0.1, 0.9, value=0.25, step=0.05, label="Confidence threshold")
            iou_sl    = gr.Slider(0.1, 0.9, value=0.45, step=0.05, label="IoU threshold (NMS)")
            run_btn   = gr.Button("Detect", variant="primary")

        with gr.Column(scale=1):
            image_out   = gr.Image(label="Detections", type="numpy")
            summary_out = gr.Markdown(label="Summary")
            table_out   = gr.Dataframe(
                headers=["Class", "Conf", "x1", "y1", "x2", "y2"],
                label="Detection Table",
                interactive=False,
            )

    run_btn.click(
        fn=detect,
        inputs=[model_dd, image_in, conf_sl, iou_sl],
        outputs=[image_out, summary_out, table_out],
    )

if __name__ == "__main__":
    demo.launch()
