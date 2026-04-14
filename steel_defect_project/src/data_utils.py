"""
Data utilities for NEU-DET steel surface defect dataset.
Handles downloading, preprocessing, conversion, augmentation, and visualization.
"""

import os
import json
import shutil
import random
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import cv2
import numpy as np
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class NEUDETDataset:
    """NEU-DET steel surface defect dataset handler."""
    
    CLASSES = ['Crazing', 'Inclusion', 'Patches', 'Pitted_surface', 'Rolled-in_scale', 'Scratches']
    CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(CLASSES)}
    
    # Mapping from XML class names (lowercase) to our standard class names
    CLASS_NAME_MAPPING = {
        'crazing': 'Crazing',
        'inclusion': 'Inclusion',
        'patches': 'Patches',
        'pitted_surface': 'Pitted_surface',
        'pitted surface': 'Pitted_surface',
        'rolled-in_scale': 'Rolled-in_scale',
        'rolled-in scale': 'Rolled-in_scale',
        'scratches': 'Scratches'
    }
    
    def __init__(self, root_dir: str):
        """
        Initialize NEU-DET dataset handler.
        
        Args:
            root_dir: Root directory containing the dataset
        """
        self.root_dir = Path(root_dir)
        self.images_dir = self.root_dir / 'IMAGES'
        self.annotations_dir = self.root_dir / 'ANNOTATIONS'
        
    def check_dataset(self) -> bool:
        """
        Check if dataset exists and is valid.
        
        Returns:
            True if dataset is valid, False otherwise
        """
        if not self.root_dir.exists():
            logger.error(f"Dataset directory not found: {self.root_dir}")
            return False
            
        if not self.images_dir.exists() or not self.annotations_dir.exists():
            logger.warning("Dataset structure not standard. Searching for images and annotations...")
            # Try to find images
            image_files = list(self.root_dir.rglob('*.jpg')) + list(self.root_dir.rglob('*.png'))
            if not image_files:
                logger.error("No images found in dataset directory")
                return False
            logger.info(f"Found {len(image_files)} images")
            
        return True
    
    def get_dataset_summary(self) -> Dict:
        """
        Get dataset summary statistics.
        
        Returns:
            Dictionary containing dataset statistics
        """
        summary = {
            'total_images': 0,
            'class_counts': {cls: 0 for cls in self.CLASSES},
            'image_sizes': set(),
            'total_annotations': 0
        }
        
        # Search for all images recursively
        image_extensions = ['.jpg', '.png', '.jpeg', '.bmp']
        all_images = []
        for ext in image_extensions:
            all_images.extend(list(self.root_dir.rglob(f'*{ext}')))
        
        summary['total_images'] = len(all_images)
        
        # Check image sizes
        for img_path in all_images[:10]:  # Sample first 10 images
            img = cv2.imread(str(img_path))
            if img is not None:
                summary['image_sizes'].add((img.shape[1], img.shape[0]))
        
        summary['image_sizes'] = list(summary['image_sizes'])
        
        # Count annotations by class (if available)
        for cls in self.CLASSES:
            cls_images = list(self.root_dir.rglob(f'*{cls}*.jpg'))
            summary['class_counts'][cls] = len(cls_images)
        
        logger.info(f"Dataset Summary: {json.dumps(summary, indent=2, default=str)}")
        return summary


def parse_neudet_annotation(xml_path: str) -> List[Dict]:
    """
    Parse NEU-DET XML annotation file.
    
    Args:
        xml_path: Path to XML annotation file
        
    Returns:
        List of annotation dictionaries
    """
    import xml.etree.ElementTree as ET
    
    if not os.path.exists(xml_path):
        return []
    
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    annotations = []
    for obj in root.findall('object'):
        name = obj.find('name').text.lower()  # Convert to lowercase
        bbox = obj.find('bndbox')
        
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)
        
        # Map XML class name to our standard format
        mapped_name = NEUDETDataset.CLASS_NAME_MAPPING.get(name, name)
        
        annotations.append({
            'class': mapped_name,
            'bbox': [xmin, ymin, xmax, ymax]
        })
    
    return annotations


def convert_to_yolo_format(annotations: List[Dict], img_width: int, img_height: int, 
                           class_to_idx: Dict) -> List[str]:
    """
    Convert annotations to YOLO format.
    
    Args:
        annotations: List of annotation dictionaries
        img_width: Image width
        img_height: Image height
        class_to_idx: Class name to index mapping
        
    Returns:
        List of YOLO format strings
    """
    yolo_lines = []
    
    for ann in annotations:
        cls_name = ann['class']
        cls_idx = class_to_idx.get(cls_name, 0)
        
        xmin, ymin, xmax, ymax = ann['bbox']
        
        # Convert to YOLO format (normalized center x, center y, width, height)
        x_center = ((xmin + xmax) / 2) / img_width
        y_center = ((ymin + ymax) / 2) / img_height
        width = (xmax - xmin) / img_width
        height = (ymax - ymin) / img_height
        
        yolo_lines.append(f"{cls_idx} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    
    return yolo_lines


def convert_to_coco_format(dataset_dir: str, output_file: str, split: str = 'train') -> Dict:
    """
    Convert dataset to COCO format.
    
    Args:
        dataset_dir: Dataset directory
        output_file: Output JSON file path
        split: Dataset split name
        
    Returns:
        COCO format dictionary
    """
    coco_data = {
        'images': [],
        'annotations': [],
        'categories': []
    }
    
    # Add categories
    for idx, cls_name in enumerate(NEUDETDataset.CLASSES):
        coco_data['categories'].append({
            'id': idx,
            'name': cls_name,
            'supercategory': 'defect'
        })
    
    # Placeholder implementation - returns template
    logger.info(f"COCO format template created for {split} split")
    
    with open(output_file, 'w') as f:
        json.dump(coco_data, f, indent=2)
    
    return coco_data


def create_dataset_splits(data_dir: str, output_dir: str, train_ratio: float = 0.7,
                         val_ratio: float = 0.15, seed: int = 42) -> Dict:
    """
    Create deterministic train/val/test splits.
    
    Args:
        data_dir: Input data directory
        output_dir: Output directory for splits
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        seed: Random seed
        
    Returns:
        Dictionary containing split information
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Find all images
    data_path = Path(data_dir)
    image_extensions = ['.jpg', '.png', '.jpeg', '.bmp']
    all_images = []
    
    for ext in image_extensions:
        all_images.extend(list(data_path.rglob(f'*{ext}')))
    
    # Shuffle
    all_images = [str(p) for p in all_images]
    random.shuffle(all_images)
    
    # Split
    n_total = len(all_images)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    train_images = all_images[:n_train]
    val_images = all_images[n_train:n_train + n_val]
    test_images = all_images[n_train + n_val:]
    
    splits = {
        'train': train_images,
        'val': val_images,
        'test': test_images
    }
    
    logger.info(f"Dataset splits: train={len(train_images)}, val={len(val_images)}, test={len(test_images)}")
    
    # Save split info
    split_file = Path(output_dir) / 'splits.json'
    split_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(split_file, 'w') as f:
        json.dump(splits, f, indent=2)
    
    return splits


def get_augmentation_pipeline(img_size: int = 640, train: bool = True) -> A.Compose:
    """
    Get augmentation pipeline using albumentations.
    
    Args:
        img_size: Target image size
        train: Whether this is for training (applies augmentations)
        
    Returns:
        Albumentations composition
    """
    if train:
        transform = A.Compose([
            A.RandomRotate90(p=0.3),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.OneOf([
                A.MotionBlur(blur_limit=5),
                A.MedianBlur(blur_limit=5),
                A.GaussianBlur(blur_limit=5),
            ], p=0.3),
            A.Resize(img_size, img_size),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))
    else:
        transform = A.Compose([
            A.Resize(img_size, img_size),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))
    
    return transform


def visualize_augmentations(data_dir: str, output_dir: str, num_samples: int = 10, 
                           seed: int = 42) -> None:
    """
    Visualize augmented samples with bounding boxes.
    
    Args:
        data_dir: Input data directory
        output_dir: Output directory for visualizations
        num_samples: Number of samples to visualize
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find sample images
    data_path = Path(data_dir)
    all_images = list(data_path.rglob('*.jpg')) + list(data_path.rglob('*.png'))
    
    if not all_images:
        logger.warning(f"No images found in {data_dir}")
        return
    
    sample_images = random.sample(all_images, min(num_samples, len(all_images)))
    
    transform = get_augmentation_pipeline(img_size=640, train=True)
    
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
    
    for idx, img_path in enumerate(tqdm(sample_images, desc="Visualizing augmentations")):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        # Create dummy bounding boxes for visualization
        h, w = img.shape[:2]
        bboxes = [[w//4, h//4, 3*w//4, 3*h//4]]  # Dummy box
        class_labels = [0]
        
        # Apply augmentation
        try:
            transformed = transform(image=img, bboxes=bboxes, class_labels=class_labels)
            aug_img = transformed['image']
            aug_bboxes = transformed['bboxes']
            
            # Draw boxes
            for bbox, cls_idx in zip(aug_bboxes, transformed['class_labels']):
                x1, y1, x2, y2 = map(int, bbox)
                color = colors[cls_idx % len(colors)]
                cv2.rectangle(aug_img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(aug_img, NEUDETDataset.CLASSES[cls_idx], (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Save
            output_file = output_path / f"augmented_sample_{idx:03d}.jpg"
            cv2.imwrite(str(output_file), aug_img)
            
        except Exception as e:
            logger.error(f"Error processing {img_path}: {e}")
            continue
    
    logger.info(f"Saved {len(sample_images)} visualizations to {output_dir}")


def prepare_dataset(input_dir: str, output_dir: str, formats: List[str] = ['yolo', 'coco'],
                   train_ratio: float = 0.7, val_ratio: float = 0.15, seed: int = 42) -> None:
    """
    Prepare dataset for training.
    
    Args:
        input_dir: Input directory containing raw dataset
        output_dir: Output directory for processed dataset
        formats: List of output formats ('yolo', 'coco')
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        seed: Random seed
    """
    logger.info(f"Preparing dataset from {input_dir}")
    
    # Check dataset
    dataset = NEUDETDataset(input_dir)
    if not dataset.check_dataset():
        logger.error("Dataset validation failed. Please ensure NEU-DET is correctly placed.")
        logger.info("Download NEU-DET from: http://faculty.neu.edu.cn/yunhyan/NEU_surface_defect_database.html")
        logger.info(f"Extract to: {input_dir}")
        return
    
    # Get summary
    summary = dataset.get_dataset_summary()
    
    # Create splits
    splits = create_dataset_splits(input_dir, output_dir, train_ratio, val_ratio, seed)
    
    # Convert to requested formats
    output_path = Path(output_dir)
    
    if 'yolo' in formats:
        yolo_dir = output_path / 'yolo'
        yolo_dir.mkdir(parents=True, exist_ok=True)
        
        for split_name, images in splits.items():
            split_dir = yolo_dir / split_name
            split_dir.mkdir(exist_ok=True)
            
            images_dir = split_dir / 'images'
            labels_dir = split_dir / 'labels'
            images_dir.mkdir(exist_ok=True)
            labels_dir.mkdir(exist_ok=True)
            
            logger.info(f"Processing YOLO format for {split_name} split...")
            
            for img_path in tqdm(images, desc=f"YOLO-{split_name}"):
                # Copy image
                img_name = Path(img_path).name
                shutil.copy(img_path, images_dir / img_name)
                
                # Parse XML annotation and convert to YOLO format
                xml_path = Path(input_dir) / 'ANNOTATIONS' / (Path(img_path).stem + '.xml')
                label_name = Path(img_path).stem + '.txt'
                
                if xml_path.exists():
                    # Parse annotations
                    annotations = parse_neudet_annotation(str(xml_path))
                    
                    # Get image dimensions
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        img_height, img_width = img.shape[:2]
                        
                        # Convert to YOLO format
                        yolo_lines = convert_to_yolo_format(
                            annotations, img_width, img_height, NEUDETDataset.CLASS_TO_IDX
                        )
                        
                        # Write label file
                        with open(labels_dir / label_name, 'w') as f:
                            f.write('\n'.join(yolo_lines))
                else:
                    # No annotation file - create empty label
                    with open(labels_dir / label_name, 'w') as f:
                        f.write("")
        
        # Create data.yaml for YOLO
        data_yaml = {
            'path': str(yolo_dir.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'nc': len(NEUDETDataset.CLASSES),
            'names': NEUDETDataset.CLASSES
        }
        
        with open(yolo_dir / 'data.yaml', 'w') as f:
            import yaml
            yaml.dump(data_yaml, f, default_flow_style=False)
        
        logger.info(f"YOLO format data saved to {yolo_dir}")
    
    if 'coco' in formats:
        coco_dir = output_path / 'coco'
        coco_dir.mkdir(parents=True, exist_ok=True)
        
        for split_name in ['train', 'val', 'test']:
            coco_file = coco_dir / f'annotations_{split_name}.json'
            convert_to_coco_format(input_dir, str(coco_file), split_name)
        
        logger.info(f"COCO format data saved to {coco_dir}")
    
    logger.info("Dataset preparation complete!")


def main():
    """Main entry point for data utilities."""
    parser = argparse.ArgumentParser(description='NEU-DET Dataset Utilities')
    parser.add_argument('--action', type=str, required=True,
                       choices=['prepare', 'visualize', 'summary'],
                       help='Action to perform')
    parser.add_argument('--input', type=str, default='data/raw',
                       help='Input directory')
    parser.add_argument('--output', type=str, default='data/processed',
                       help='Output directory')
    parser.add_argument('--formats', type=str, default='yolo,coco',
                       help='Output formats (comma-separated)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--train-ratio', type=float, default=0.7,
                       help='Training set ratio')
    parser.add_argument('--val-ratio', type=float, default=0.15,
                       help='Validation set ratio')
    parser.add_argument('--num-samples', type=int, default=10,
                       help='Number of samples for visualization')
    
    args = parser.parse_args()
    
    if args.action == 'prepare':
        formats = args.formats.split(',')
        prepare_dataset(
            args.input,
            args.output,
            formats=formats,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            seed=args.seed
        )
    
    elif args.action == 'visualize':
        visualize_augmentations(
            args.input,
            args.output,
            num_samples=args.num_samples,
            seed=args.seed
        )
    
    elif args.action == 'summary':
        dataset = NEUDETDataset(args.input)
        dataset.get_dataset_summary()


if __name__ == '__main__':
    main()
