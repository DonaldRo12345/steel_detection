---
title: Steel Surface Defect Detection
emoji: 🔩
colorFrom: gray
colorTo: red
sdk: gradio
sdk_version: "4.0.0"
app_file: app.py
pinned: false
license: mit
---

# Steel Surface Defect Detection

Object detection on steel surface images using models trained on the [NEU-DET dataset](http://faculty.neu.edu.cn/yunhyan/NEU_surface_defect_database.html).

## Models

| Model | mAP@0.5 | Description |
|---|---|---|
| YOLOv8n Baseline | **0.741** | 50 epochs, standard training |
| Enhanced YOLOv8 (CLAHE) | 0.664 | 30 epochs, CLAHE contrast enhancement |
| DETR-Lite | 0.101 | Lightweight transformer, 15 epochs |

## Defect Classes
Crazing · Inclusion · Patches · Pitted Surface · Rolled-in Scale · Scratches
