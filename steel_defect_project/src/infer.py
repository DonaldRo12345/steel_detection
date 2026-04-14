"""
Inference script for steel defect detection.
Run detection on single image or batch of images.
"""

import os
import argparse
import logging
from pathlib import Path
from typing import List, Union
import cv2
import numpy as np
import torch
from tqdm import tqdm

from utils import set_seed, get_device


logger = logging.getLogger(__name__)


# Color palette for visualization
COLORS = [
    (255, 0, 0),      # Red
    (0, 255, 0),      # Green
    (0, 0, 255),      # Blue
    (255, 255, 0),    # Yellow
    (255, 0, 255),    # Magenta
    (0, 255, 255),    # Cyan
]

CLASS_NAMES = ['Crazing', 'Inclusion', 'Patches', 'Pitted_surface', 'Rolled-in_scale', 'Scratches']


def draw_detections(image: np.ndarray, boxes: np.ndarray, scores: np.ndarray, 
                   classes: np.ndarray, class_names: List[str] = None) -> np.ndarray:
    """
    Draw bounding boxes on image.
    
    Args:
        image: Input image
        boxes: Bounding boxes [N, 4] in xyxy format
        scores: Confidence scores [N]
        classes: Class indices [N]
        class_names: List of class names
        
    Returns:
        Image with drawn boxes
    """
    if class_names is None:
        class_names = CLASS_NAMES
    
    img_draw = image.copy()
    
    for box, score, cls in zip(boxes, scores, classes):
        x1, y1, x2, y2 = map(int, box)
        cls_idx = int(cls)
        
        # Select color
        color = COLORS[cls_idx % len(COLORS)]
        
        # Draw box
        cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        label = f"{class_names[cls_idx]}: {score:.2f}"
        
        # Get label size
        (label_w, label_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        
        # Draw label background
        cv2.rectangle(img_draw, (x1, y1 - label_h - baseline - 5), 
                     (x1 + label_w, y1), color, -1)
        
        # Draw label text
        cv2.putText(img_draw, label, (x1, y1 - baseline - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return img_draw


def infer_yolo(model_path: str, image_path: str, device: str,
              conf_thres: float = 0.25, iou_thres: float = 0.45) -> dict:
    """
    Run YOLO inference on single image.
    
    Args:
        model_path: Path to YOLO model
        image_path: Path to input image
        device: Device to use
        conf_thres: Confidence threshold
        iou_thres: IoU threshold for NMS
        
    Returns:
        Dictionary with detection results
    """
    from ultralytics import YOLO
    
    model = YOLO(model_path)
    
    results = model(image_path, device=device, conf=conf_thres, 
                   iou=iou_thres, verbose=False)
    
    result = results[0]
    
    # Extract detections
    boxes = result.boxes.xyxy.cpu().numpy()
    scores = result.boxes.conf.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy()
    
    return {
        'boxes': boxes,
        'scores': scores,
        'classes': classes,
        'num_detections': len(boxes)
    }


def infer_detr(model_path: str, image_path: str, device: torch.device,
              conf_thres: float = 0.5) -> dict:
    """
    Run DETR inference on single image.
    
    Args:
        model_path: Path to DETR model checkpoint
        image_path: Path to input image
        device: Device to use
        conf_thres: Confidence threshold
        
    Returns:
        Dictionary with detection results
    """
    from train_detr import DETRLite
    import torchvision.transforms as T
    
    # Load model
    model = DETRLite(num_classes=6)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Load and preprocess image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]
    
    transform = T.Compose([
        T.ToTensor(),
        T.Resize((640, 640)),
    ])
    
    img_tensor = transform(image_rgb).unsqueeze(0).to(device)
    
    # Inference
    with torch.no_grad():
        outputs = model(img_tensor)
    
    # Post-process
    logits = outputs['pred_logits'][0]  # [num_queries, num_classes+1]
    boxes = outputs['pred_boxes'][0]    # [num_queries, 4]
    
    # Get predictions with confidence > threshold
    probs = logits.softmax(-1)
    scores, labels = probs[:, :-1].max(-1)  # Exclude no-object class
    
    keep = scores > conf_thres
    scores = scores[keep].cpu().numpy()
    labels = labels[keep].cpu().numpy()
    boxes = boxes[keep].cpu().numpy()
    
    # Convert boxes from normalized to pixel coordinates
    boxes_xyxy = np.zeros_like(boxes)
    boxes_xyxy[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2) * w  # x1
    boxes_xyxy[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2) * h  # y1
    boxes_xyxy[:, 2] = (boxes[:, 0] + boxes[:, 2] / 2) * w  # x2
    boxes_xyxy[:, 3] = (boxes[:, 1] + boxes[:, 3] / 2) * h  # y2
    
    return {
        'boxes': boxes_xyxy,
        'scores': scores,
        'classes': labels,
        'num_detections': len(boxes_xyxy)
    }


def infer_fasterrcnn(model_path: str, image_path: str, device: torch.device,
                     conf_thres: float = 0.5) -> dict:
    """Run Faster R-CNN inference on a single image."""
    from train_fasterrcnn import build_fasterrcnn, CLASS_NAMES

    num_classes = len(CLASS_NAMES) + 1
    model = build_fasterrcnn(num_classes=num_classes, pretrained=False)
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()

    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (640, 640))
    img_tensor = torch.as_tensor(img_resized, dtype=torch.float32).permute(2, 0, 1) / 255.0
    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        outputs = model([img_tensor])[0]

    keep = outputs['scores'] > conf_thres
    boxes = outputs['boxes'][keep].cpu().numpy()
    scores = outputs['scores'][keep].cpu().numpy()
    # labels from Faster R-CNN are 1-indexed (0 = bg); map to 0-indexed
    classes = (outputs['labels'][keep] - 1).clamp(min=0).cpu().numpy()

    # Scale boxes back to original image size
    h, w = img.shape[:2]
    boxes[:, [0, 2]] *= w / 640
    boxes[:, [1, 3]] *= h / 640

    return {
        'boxes': boxes,
        'scores': scores,
        'classes': classes,
        'num_detections': len(boxes),
    }


def infer_rtdetr(model_path: str, image_path: str, device: str,
                 conf_thres: float = 0.25, iou_thres: float = 0.45) -> dict:
    """Run RT-DETR inference on a single image (Ultralytics API)."""
    from ultralytics import RTDETR

    model = RTDETR(model_path)
    results = model(image_path, device=device, conf=conf_thres,
                    iou=iou_thres, verbose=False)
    result = results[0]
    return {
        'boxes': result.boxes.xyxy.cpu().numpy(),
        'scores': result.boxes.conf.cpu().numpy(),
        'classes': result.boxes.cls.cpu().numpy(),
        'num_detections': len(result.boxes),
    }


def process_single_image(model_path: str, image_path: str, output_path: str,
                        model_type: str = 'yolo', device: str = None,
                        conf_thres: float = 0.25, iou_thres: float = 0.45) -> dict:
    """
    Process single image.
    
    Args:
        model_path: Path to model checkpoint
        image_path: Path to input image
        output_path: Path to save output image
        model_type: 'yolo' or 'detr'
        device: Device to use
        conf_thres: Confidence threshold
        iou_thres: IoU threshold (for YOLO)
        
    Returns:
        Detection results dictionary
    """
    device_obj = get_device(device)
    
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        logger.error(f"Failed to read image: {image_path}")
        return None
    
    # Run inference
    if model_type in ('yolo', 'yolo_enhanced'):
        results = infer_yolo(model_path, image_path, str(device_obj), 
                           conf_thres, iou_thres)
    elif model_type == 'fasterrcnn':
        results = infer_fasterrcnn(model_path, image_path, device_obj, conf_thres)
    elif model_type == 'rtdetr':
        results = infer_rtdetr(model_path, image_path, str(device_obj),
                              conf_thres, iou_thres)
    else:
        results = infer_detr(model_path, image_path, device_obj, conf_thres)
    
    # Draw detections
    if results['num_detections'] > 0:
        image_out = draw_detections(
            image, 
            results['boxes'], 
            results['scores'], 
            results['classes']
        )
    else:
        image_out = image
        logger.info(f"No detections found in {image_path}")
    
    # Save output
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_file), image_out)
    
    logger.info(f"Output saved to {output_path}")
    logger.info(f"Detections: {results['num_detections']}")
    
    return results


def process_directory(model_path: str, input_dir: str, output_dir: str,
                     model_type: str = 'yolo', device: str = None,
                     conf_thres: float = 0.25, iou_thres: float = 0.45) -> List[dict]:
    """
    Process all images in a directory. Loads the model once for efficiency.
    
    Args:
        model_path: Path to model checkpoint
        input_dir: Input directory containing images
        output_dir: Output directory for results
        model_type: 'yolo' or 'detr'
        device: Device to use
        conf_thres: Confidence threshold
        iou_thres: IoU threshold (for YOLO)
        
    Returns:
        List of detection results
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Find all images
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(input_path.glob(f'*{ext}')))
        image_files.extend(list(input_path.glob(f'*{ext.upper()}')))

    if not image_files:
        logger.warning(f"No images found in {input_dir}")
        return []

    logger.info(f"Found {len(image_files)} images")

    device_obj = get_device(device)

    if model_type == 'yolo':
        # Load model ONCE, then batch predict
        from ultralytics import YOLO
        logger.info(f"Loading YOLO model from {model_path}")
        model = YOLO(model_path)

        all_results = []
        for img_file in tqdm(image_files, desc="Processing images"):
            results_raw = model(
                str(img_file),
                device=str(device_obj),
                conf=conf_thres,
                iou=iou_thres,
                verbose=False
            )
            result = results_raw[0]
            boxes  = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()

            image = cv2.imread(str(img_file))
            if len(boxes) > 0:
                image_out = draw_detections(image, boxes, scores, classes)
            else:
                image_out = image
                logger.debug(f"No detections: {img_file.name}")

            out_file = output_path / f"{img_file.stem}_detected{img_file.suffix}"
            cv2.imwrite(str(out_file), image_out)

            all_results.append({
                'image': str(img_file),
                'boxes': boxes,
                'scores': scores,
                'classes': classes,
                'num_detections': len(boxes)
            })

    else:
        # DETR: load model once
        from train_detr import DETRLite
        logger.info(f"Loading DETR model from {model_path}")
        model = DETRLite(num_classes=6)
        checkpoint = torch.load(model_path, map_location=device_obj)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device_obj)
        model.eval()

        all_results = []
        for img_file in tqdm(image_files, desc="Processing images"):
            output_file = output_path / f"{img_file.stem}_detected{img_file.suffix}"
            results = process_single_image(
                model_path, str(img_file), str(output_file),
                model_type, str(device_obj), conf_thres, iou_thres
            )
            if results:
                results['image'] = str(img_file)
                all_results.append(results)

    logger.info(f"Processed {len(all_results)} images successfully")
    return all_results


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run inference for steel defect detection')
    
    parser.add_argument('--model', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--model-type', type=str, default='yolo',
                       choices=['yolo', 'detr', 'fasterrcnn', 'rtdetr', 'yolo_enhanced'],
                       help='Model type')
    parser.add_argument('--image', type=str, default=None,
                       help='Path to input image')
    parser.add_argument('--dir', type=str, default=None,
                       help='Path to input directory')
    parser.add_argument('--out-dir', type=str, default='results/visuals',
                       help='Output directory')
    parser.add_argument('--device', type=str, default=None,
                       help='Device (cuda:0, cpu, etc.)')
    parser.add_argument('--conf-thres', type=float, default=0.25,
                       help='Confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45,
                       help='IoU threshold for NMS (YOLO only)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    return parser.parse_args()


def main():
    """Main inference entry point."""
    args = parse_args()
    
    # Setup
    set_seed(args.seed)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("="*80)
    logger.info("Steel Defect Detection Inference")
    logger.info("="*80)
    
    if args.image:
        # Process single image
        output_file = Path(args.out_dir) / f"{Path(args.image).stem}_detected.jpg"
        process_single_image(
            args.model, args.image, str(output_file),
            args.model_type, args.device, args.conf_thres, args.iou_thres
        )
    
    elif args.dir:
        # Process directory
        process_directory(
            args.model, args.dir, args.out_dir,
            args.model_type, args.device, args.conf_thres, args.iou_thres
        )
    
    else:
        logger.error("Please specify either --image or --dir")
        return
    
    logger.info("="*80)
    logger.info("Inference complete!")
    logger.info("="*80)


if __name__ == '__main__':
    main()
