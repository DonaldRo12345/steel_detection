"""
Model optimization and export utilities.
Supports ONNX export and quantization.
"""

import os
import argparse
import logging
from pathlib import Path
import torch
import torch.quantization
from ultralytics import YOLO

from utils import get_device


logger = logging.getLogger(__name__)


def export_yolo_to_onnx(model_path: str, output_path: str, img_size: int = 640, 
                       simplify: bool = True) -> str:
    """
    Export YOLO model to ONNX format.
    
    Args:
        model_path: Path to YOLO model
        output_path: Output path for ONNX model
        img_size: Input image size
        simplify: Whether to simplify the ONNX model
        
    Returns:
        Path to exported ONNX model
    """
    logger.info(f"Loading YOLO model from {model_path}")
    model = YOLO(model_path)
    
    # Export to ONNX
    logger.info(f"Exporting to ONNX (img_size={img_size}, simplify={simplify})")
    
    export_path = model.export(
        format='onnx',
        imgsz=img_size,
        simplify=simplify,
        opset=12
    )
    
    # Move to desired location if different
    if export_path != output_path:
        import shutil
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(export_path, output_path)
        logger.info(f"ONNX model saved to {output_path}")
    
    return output_path


def export_detr_to_onnx(model_path: str, output_path: str, img_size: int = 640) -> str:
    """
    Export DETR model to ONNX format.
    
    Args:
        model_path: Path to DETR checkpoint
        output_path: Output path for ONNX model
        img_size: Input image size
        
    Returns:
        Path to exported ONNX model
    """
    from train_detr import DETRLite
    
    logger.info(f"Loading DETR model from {model_path}")
    
    model = DETRLite(num_classes=6)
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, img_size, img_size)
    
    # Export
    logger.info(f"Exporting DETR to ONNX (img_size={img_size})")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['images'],
        output_names=['pred_logits', 'pred_boxes'],
        dynamic_axes={
            'images': {0: 'batch_size'},
            'pred_logits': {0: 'batch_size'},
            'pred_boxes': {0: 'batch_size'}
        }
    )
    
    logger.info(f"ONNX model saved to {output_path}")
    return output_path


def quantize_pytorch_model(model_path: str, output_path: str, model_type: str = 'detr') -> str:
    """
    Apply dynamic quantization to PyTorch model.
    
    Args:
        model_path: Path to model checkpoint
        output_path: Output path for quantized model
        model_type: 'yolo' or 'detr'
        
    Returns:
        Path to quantized model
    """
    logger.info(f"Loading model from {model_path}")
    
    if model_type == 'yolo':
        logger.warning("YOLO quantization via Ultralytics - use built-in export")
        model = YOLO(model_path)
        
        # Export int8 quantized model
        export_path = model.export(format='torchscript', int8=True)
        
        import shutil
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(export_path, output_path)
        
    else:  # DETR
        from train_detr import DETRLite
        
        model = DETRLite(num_classes=6)
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Apply dynamic quantization
        logger.info("Applying dynamic quantization...")
        quantized_model = torch.quantization.quantize_dynamic(
            model,
            {torch.nn.Linear, torch.nn.Conv2d},
            dtype=torch.qint8
        )
        
        # Save quantized model
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': quantized_model.state_dict(),
            'quantized': True
        }, output_path)
    
    logger.info(f"Quantized model saved to {output_path}")
    
    # Compare model sizes
    original_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
    quantized_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
    
    logger.info(f"Original model size: {original_size:.2f} MB")
    logger.info(f"Quantized model size: {quantized_size:.2f} MB")
    logger.info(f"Size reduction: {(1 - quantized_size/original_size)*100:.1f}%")
    
    return output_path


def benchmark_onnx_model(onnx_path: str, img_size: int = 640, 
                        num_iterations: int = 100) -> dict:
    """
    Benchmark ONNX model inference speed.
    
    Args:
        onnx_path: Path to ONNX model
        img_size: Input image size
        num_iterations: Number of iterations for benchmarking
        
    Returns:
        Dictionary with benchmark results
    """
    import time
    import numpy as np
    
    try:
        import onnxruntime as ort
    except ImportError:
        logger.error("onnxruntime not installed. Install with: pip install onnxruntime")
        return {}
    
    logger.info(f"Benchmarking ONNX model: {onnx_path}")
    
    # Create session
    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    
    # Get input name
    input_name = session.get_inputs()[0].name
    
    # Create dummy input
    dummy_input = np.random.randn(1, 3, img_size, img_size).astype(np.float32)
    
    # Warmup
    for _ in range(10):
        _ = session.run(None, {input_name: dummy_input})
    
    # Benchmark
    times = []
    for _ in range(num_iterations):
        start = time.time()
        _ = session.run(None, {input_name: dummy_input})
        times.append(time.time() - start)
    
    times = np.array(times)
    
    results = {
        'mean_latency_ms': float(times.mean() * 1000),
        'std_latency_ms': float(times.std() * 1000),
        'fps': float(1.0 / times.mean()),
        'num_iterations': num_iterations
    }
    
    logger.info(f"ONNX Inference - FPS: {results['fps']:.2f}, "
               f"Latency: {results['mean_latency_ms']:.2f} ± {results['std_latency_ms']:.2f} ms")
    
    return results


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Model optimization and export')
    
    parser.add_argument('--model', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--model-type', type=str, default='yolo',
                       choices=['yolo', 'detr'],
                       help='Model type')
    parser.add_argument('--action', type=str, required=True,
                       choices=['export-onnx', 'quantize', 'benchmark'],
                       help='Action to perform')
    parser.add_argument('--output-dir', type=str, default='results/models',
                       help='Output directory')
    parser.add_argument('--img-size', type=int, default=640,
                       help='Input image size')
    parser.add_argument('--simplify', action='store_true',
                       help='Simplify ONNX model (YOLO only)')
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("="*80)
    logger.info("Model Optimization and Export")
    logger.info("="*80)
    
    output_dir = Path(args.output_dir)
    
    if args.action == 'export-onnx':
        onnx_dir = output_dir / 'onnx'
        onnx_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = onnx_dir / f"{args.model_type}_model.onnx"
        
        if args.model_type == 'yolo':
            export_yolo_to_onnx(args.model, str(output_file), args.img_size, args.simplify)
        else:
            export_detr_to_onnx(args.model, str(output_file), args.img_size)
    
    elif args.action == 'quantize':
        quant_dir = output_dir / 'quantized'
        quant_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = quant_dir / f"{args.model_type}_quantized.pth"
        quantize_pytorch_model(args.model, str(output_file), args.model_type)
    
    elif args.action == 'benchmark':
        # Assume model is already in ONNX format
        results = benchmark_onnx_model(args.model, args.img_size)
        print(f"Benchmark results: {results}")
    
    logger.info("="*80)
    logger.info("Operation complete!")
    logger.info("="*80)


if __name__ == '__main__':
    main()
