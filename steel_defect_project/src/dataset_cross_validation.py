"""
Multi-dataset utilities for cross-validation on industrial defect benchmarks.

Supported datasets
------------------
1. **NEU-DET** — 1,800 balanced grayscale images, 6 classes (already in project).
2. **GC10-DET** — 3,570 real-world galvanized steel images, 10 classes, naturally
   imbalanced (punching, welding line, crescent gap, water spot, oil spot,
   silk spot, inclusion, rolled pit, crease, waist folding).
3. **X-SDD**   — 1,360 strip-steel images, 7 classes, highly imbalanced and
   includes very small defects.

Each dataset class normalises annotations into a unified format compatible
with YOLO and COCO pipelines so that the same training scripts can be reused.

Usage
-----
    python src/dataset_cross_validation.py \\
        --datasets neu gc10 xsdd \\
        --raw-dir data/raw \\
        --output-dir data/processed \\
        --formats yolo,coco \\
        --seed 42
"""

import os
import json
import shutil
import random
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import Counter

import cv2
import numpy as np
from tqdm import tqdm
import yaml

logger = logging.getLogger(__name__)

# ═════════════════════════════════════════════════════════════════════════════
#  Dataset definitions
# ═════════════════════════════════════════════════════════════════════════════

NEU_CLASSES = ['Crazing', 'Inclusion', 'Patches', 'Pitted_surface',
               'Rolled-in_scale', 'Scratches']

GC10_CLASSES = ['Punching', 'Welding_line', 'Crescent_gap', 'Water_spot',
                'Oil_spot', 'Silk_spot', 'Inclusion', 'Rolled_pit',
                'Crease', 'Waist_folding']

XSDD_CLASSES = ['Crazing', 'Inclusion', 'Patches', 'Pitted_surface',
                'Rolled-in_scale', 'Scratches', 'Edge_crack']


# ── Unified annotation container ─────────────────────────────────────────────

class UnifiedAnnotation:
    """Simple container for one annotated image."""

    def __init__(self, image_path: str, boxes: List[List[int]],
                 labels: List[str], img_w: int, img_h: int):
        self.image_path = image_path
        self.boxes = boxes          # list of [xmin, ymin, xmax, ymax]
        self.labels = labels        # list of class-name strings
        self.img_w = img_w
        self.img_h = img_h


# ═════════════════════════════════════════════════════════════════════════════
#  Dataset parsers
# ═════════════════════════════════════════════════════════════════════════════

def parse_neu_det(raw_dir: str) -> Tuple[List[UnifiedAnnotation], List[str]]:
    """Parse existing NEU-DET dataset."""
    import xml.etree.ElementTree as ET

    dataset_dir = Path(raw_dir) / 'NEU-DET'
    images_dir = dataset_dir / 'IMAGES'
    annots_dir = dataset_dir / 'ANNOTATIONS'

    if not images_dir.exists():
        logger.warning(f"NEU-DET images not found at {images_dir}")
        return [], NEU_CLASSES

    samples: List[UnifiedAnnotation] = []
    for img_path in sorted(images_dir.glob('*.jpg')):
        xml_path = annots_dir / (img_path.stem + '.xml')
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        boxes, labels = [], []

        if xml_path.exists():
            tree = ET.parse(str(xml_path))
            for obj in tree.getroot().findall('object'):
                name = obj.find('name').text.lower().replace(' ', '_')
                # Map to canonical NEU name
                name_map = {
                    'crazing': 'Crazing', 'inclusion': 'Inclusion',
                    'patches': 'Patches', 'pitted_surface': 'Pitted_surface',
                    'rolled-in_scale': 'Rolled-in_scale', 'scratches': 'Scratches',
                }
                cls = name_map.get(name, name.title())
                if cls not in NEU_CLASSES:
                    continue
                bb = obj.find('bndbox')
                boxes.append([int(bb.find(t).text) for t in ('xmin', 'ymin', 'xmax', 'ymax')])
                labels.append(cls)

        samples.append(UnifiedAnnotation(str(img_path), boxes, labels, w, h))

    logger.info(f"NEU-DET: loaded {len(samples)} images")
    return samples, NEU_CLASSES


def parse_gc10_det(raw_dir: str) -> Tuple[List[UnifiedAnnotation], List[str]]:
    """Parse GC10-DET dataset.

    Expected layout
    ---------------
        raw_dir/GC10-DET/
            images/               ← .jpg files, named by numeric id
            labels/               ← YOLO-format .txt files **or** XML
    If YOLO .txt annotations exist they are preferred (class x_c y_c w h).
    """
    dataset_dir = Path(raw_dir) / 'GC10-DET'
    images_dir = dataset_dir / 'images'
    labels_dir = dataset_dir / 'labels'

    if not images_dir.exists():
        logger.warning(f"GC10-DET not found at {images_dir}. "
                       "Download from: https://github.com/lvxiaoming2019/GC10-DET")
        return [], GC10_CLASSES

    samples: List[UnifiedAnnotation] = []
    for img_path in sorted(images_dir.glob('*.*')):
        if img_path.suffix.lower() not in ('.jpg', '.jpeg', '.png', '.bmp'):
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        boxes, labels = [], []

        txt_path = labels_dir / (img_path.stem + '.txt')
        if txt_path.exists():
            with open(txt_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cls_id = int(parts[0])
                    if cls_id >= len(GC10_CLASSES):
                        continue
                    xc, yc, bw, bh = [float(x) for x in parts[1:5]]
                    x1 = int((xc - bw / 2) * w)
                    y1 = int((yc - bh / 2) * h)
                    x2 = int((xc + bw / 2) * w)
                    y2 = int((yc + bh / 2) * h)
                    boxes.append([x1, y1, x2, y2])
                    labels.append(GC10_CLASSES[cls_id])

        samples.append(UnifiedAnnotation(str(img_path), boxes, labels, w, h))

    logger.info(f"GC10-DET: loaded {len(samples)} images")
    return samples, GC10_CLASSES


def parse_xsdd(raw_dir: str) -> Tuple[List[UnifiedAnnotation], List[str]]:
    """Parse X-SDD (extended steel defect dataset).

    Expected layout
    ---------------
        raw_dir/X-SDD/
            images/
            annotations/          ← XML (Pascal VOC)  **or**  labels/ (YOLO txt)
    """
    import xml.etree.ElementTree as ET

    dataset_dir = Path(raw_dir) / 'X-SDD'
    images_dir = dataset_dir / 'images'
    annots_dir = dataset_dir / 'annotations'
    labels_dir = dataset_dir / 'labels'

    if not images_dir.exists():
        logger.warning(f"X-SDD not found at {images_dir}. "
                       "Download from appropriate academic repository.")
        return [], XSDD_CLASSES

    samples: List[UnifiedAnnotation] = []

    for img_path in sorted(images_dir.glob('*.*')):
        if img_path.suffix.lower() not in ('.jpg', '.jpeg', '.png', '.bmp'):
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        boxes, labels = [], []

        xml_path = annots_dir / (img_path.stem + '.xml') if annots_dir.exists() else None
        txt_path = labels_dir / (img_path.stem + '.txt') if labels_dir.exists() else None

        if xml_path and xml_path.exists():
            tree = ET.parse(str(xml_path))
            for obj in tree.getroot().findall('object'):
                name = obj.find('name').text.strip()
                # Attempt to match known class
                matched = None
                for c in XSDD_CLASSES:
                    if c.lower().replace('_', '') == name.lower().replace('_', '').replace(' ', ''):
                        matched = c
                        break
                if matched is None:
                    continue
                bb = obj.find('bndbox')
                boxes.append([int(bb.find(t).text) for t in ('xmin', 'ymin', 'xmax', 'ymax')])
                labels.append(matched)

        elif txt_path and txt_path.exists():
            with open(txt_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cls_id = int(parts[0])
                    if cls_id >= len(XSDD_CLASSES):
                        continue
                    xc, yc, bw, bh = [float(x) for x in parts[1:5]]
                    x1 = int((xc - bw / 2) * w)
                    y1 = int((yc - bh / 2) * h)
                    x2 = int((xc + bw / 2) * w)
                    y2 = int((yc + bh / 2) * h)
                    boxes.append([x1, y1, x2, y2])
                    labels.append(XSDD_CLASSES[cls_id])

        samples.append(UnifiedAnnotation(str(img_path), boxes, labels, w, h))

    logger.info(f"X-SDD: loaded {len(samples)} images")
    return samples, XSDD_CLASSES


DATASET_PARSERS = {
    'neu': parse_neu_det,
    'gc10': parse_gc10_det,
    'xsdd': parse_xsdd,
}


# ═════════════════════════════════════════════════════════════════════════════
#  Converting unified annotations → YOLO / COCO
# ═════════════════════════════════════════════════════════════════════════════

def export_yolo(samples: List[UnifiedAnnotation], classes: List[str],
                output_dir: Path, split_name: str, img_size: int = 640):
    """Copy images and write YOLO label files."""
    img_dir = output_dir / split_name / 'images'
    lbl_dir = output_dir / split_name / 'labels'
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    cls2id = {c: i for i, c in enumerate(classes)}

    for s in tqdm(samples, desc=f'YOLO-{split_name}', leave=False):
        dst_img = img_dir / Path(s.image_path).name
        if not dst_img.exists():
            shutil.copy(s.image_path, dst_img)

        lines = []
        for box, lbl in zip(s.boxes, s.labels):
            if lbl not in cls2id:
                continue
            xmin, ymin, xmax, ymax = box
            xc = ((xmin + xmax) / 2) / s.img_w
            yc = ((ymin + ymax) / 2) / s.img_h
            bw = (xmax - xmin) / s.img_w
            bh = (ymax - ymin) / s.img_h
            lines.append(f"{cls2id[lbl]} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

        lbl_file = lbl_dir / (Path(s.image_path).stem + '.txt')
        lbl_file.write_text('\n'.join(lines))


def export_coco(samples: List[UnifiedAnnotation], classes: List[str],
                output_file: Path, split_name: str):
    """Write COCO-format annotation JSON."""
    coco = {
        'images': [],
        'annotations': [],
        'categories': [{'id': i, 'name': c, 'supercategory': 'defect'}
                       for i, c in enumerate(classes)],
    }
    cls2id = {c: i for i, c in enumerate(classes)}
    ann_id = 1

    for img_id, s in enumerate(samples, start=1):
        coco['images'].append({
            'id': img_id,
            'file_name': Path(s.image_path).name,
            'width': s.img_w,
            'height': s.img_h,
        })
        for box, lbl in zip(s.boxes, s.labels):
            if lbl not in cls2id:
                continue
            x, y, x2, y2 = box
            w, h = x2 - x, y2 - y
            coco['annotations'].append({
                'id': ann_id,
                'image_id': img_id,
                'category_id': cls2id[lbl],
                'bbox': [x, y, w, h],
                'area': w * h,
                'iscrowd': 0,
            })
            ann_id += 1

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(coco, f, indent=2)

    logger.info(f"  COCO annotations → {output_file}  ({len(coco['annotations'])} annots)")


# ═════════════════════════════════════════════════════════════════════════════
#  Split + export
# ═════════════════════════════════════════════════════════════════════════════

def split_samples(samples: List[UnifiedAnnotation],
                  train_ratio: float = 0.7, val_ratio: float = 0.15,
                  seed: int = 42) -> Dict[str, List[UnifiedAnnotation]]:
    rng = random.Random(seed)
    idxs = list(range(len(samples)))
    rng.shuffle(idxs)

    n_train = int(len(samples) * train_ratio)
    n_val = int(len(samples) * val_ratio)

    return {
        'train': [samples[i] for i in idxs[:n_train]],
        'val': [samples[i] for i in idxs[n_train:n_train + n_val]],
        'test': [samples[i] for i in idxs[n_train + n_val:]],
    }


def print_dataset_stats(name: str, samples: List[UnifiedAnnotation], classes: List[str]):
    total_boxes = sum(len(s.boxes) for s in samples)
    cls_counts = Counter(lbl for s in samples for lbl in s.labels)
    logger.info(f"\n{'─' * 60}")
    logger.info(f"Dataset: {name}  |  Images: {len(samples)}  |  Annotations: {total_boxes}")
    for c in classes:
        logger.info(f"  {c:25s}: {cls_counts.get(c, 0):5d}")
    if cls_counts:
        max_c, min_c = max(cls_counts.values()), min(cls_counts.values())
        logger.info(f"  Imbalance ratio (max/min): {max_c / max(min_c, 1):.1f}×")
    logger.info(f"{'─' * 60}")


def prepare_single_dataset(name: str, raw_dir: str, output_dir: str,
                           formats: List[str], seed: int = 42):
    """Parse, split, and export a single dataset."""
    parser = DATASET_PARSERS.get(name)
    if parser is None:
        logger.error(f"Unknown dataset: {name}")
        return

    samples, classes = parser(raw_dir)
    if not samples:
        logger.warning(f"No samples loaded for {name}, skipping.")
        return

    print_dataset_stats(name, samples, classes)

    splits = split_samples(samples, seed=seed)
    out = Path(output_dir) / name

    if 'yolo' in formats:
        yolo_dir = out / 'yolo'
        for split_name, split_samples_list in splits.items():
            export_yolo(split_samples_list, classes, yolo_dir, split_name)
        # Write data.yaml
        data_yaml = {
            'path': str(yolo_dir.resolve()),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'nc': len(classes),
            'names': classes,
        }
        with open(yolo_dir / 'data.yaml', 'w') as f:
            yaml.dump(data_yaml, f, default_flow_style=False)
        logger.info(f"  YOLO format → {yolo_dir}")

    if 'coco' in formats:
        coco_dir = out / 'coco'
        for split_name, split_samples_list in splits.items():
            export_coco(split_samples_list, classes,
                        coco_dir / f'annotations_{split_name}.json', split_name)
        logger.info(f"  COCO format → {coco_dir}")

    # Save splits.json
    splits_json = {k: [s.image_path for s in v] for k, v in splits.items()}
    with open(out / 'splits.json', 'w') as f:
        json.dump(splits_json, f, indent=2)


# ═════════════════════════════════════════════════════════════════════════════
#  CLI
# ═════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description='Prepare multi-dataset benchmarks for cross-validation')
    p.add_argument('--datasets', nargs='+', default=['neu', 'gc10', 'xsdd'],
                   choices=list(DATASET_PARSERS.keys()),
                   help='Datasets to prepare')
    p.add_argument('--raw-dir', default='data/raw',
                   help='Root of raw dataset folders')
    p.add_argument('--output-dir', default='data/processed',
                   help='Output directory')
    p.add_argument('--formats', default='yolo,coco',
                   help='Export formats (comma-separated)')
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


def main():
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    args = parse_args()
    formats = [f.strip() for f in args.formats.split(',')]

    logger.info("=" * 80)
    logger.info("Multi-Dataset Preparation for Cross-Validation")
    logger.info(f"Datasets: {args.datasets}")
    logger.info("=" * 80)

    for ds_name in args.datasets:
        logger.info(f"\nProcessing dataset: {ds_name}")
        prepare_single_dataset(ds_name, args.raw_dir, args.output_dir,
                               formats, args.seed)

    logger.info("\n" + "=" * 80)
    logger.info("All datasets prepared!")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
