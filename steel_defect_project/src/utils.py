"""
Common utility functions for the steel defect detection project.
"""

import os
import json
import random
import logging
import platform
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import torch
import yaml


logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Make CUDA operations deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    logger.info(f"Random seed set to {seed}")


def get_device(device: Optional[str] = None) -> torch.device:
    """
    Get PyTorch device.
    
    Args:
        device: Device string (cuda, cpu, or cuda:0, etc.)
        
    Returns:
        PyTorch device object
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    device = torch.device(device)
    logger.info(f"Using device: {device}")
    
    if device.type == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    return device


def save_environment_info(output_file: str) -> Dict[str, Any]:
    """
    Save environment information for reproducibility.
    
    Args:
        output_file: Path to output JSON file
        
    Returns:
        Dictionary containing environment info
    """
    import sys
    
    env_info = {
        'python_version': sys.version,
        'platform': platform.platform(),
        'pytorch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
        'cudnn_version': torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    
    if torch.cuda.is_available():
        env_info['gpu_name'] = torch.cuda.get_device_name(0)
    
    # Save to file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(env_info, f, indent=2)
    
    logger.info(f"Environment info saved to {output_file}")
    return env_info


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Loaded config from {config_path}")
    return config


def save_config(config: Dict[str, Any], output_path: str) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        output_path: Path to output YAML file
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    logger.info(f"Config saved to {output_path}")


def save_metrics(metrics: Dict[str, Any], output_file: str) -> None:
    """
    Save metrics to JSON file.
    
    Args:
        metrics: Metrics dictionary
        output_file: Output file path
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(metrics, f, indent=2, default=str)
    
    logger.info(f"Metrics saved to {output_file}")


def load_metrics(metrics_file: str) -> Dict[str, Any]:
    """
    Load metrics from JSON file.
    
    Args:
        metrics_file: Path to metrics JSON file
        
    Returns:
        Metrics dictionary
    """
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
    
    return metrics


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self, name: str = ''):
        self.name = name
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def setup_logging(log_dir: str, log_file: str = 'training.log') -> None:
    """
    Setup logging configuration.
    
    Args:
        log_dir: Directory for log files
        log_file: Log file name
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    log_file_path = log_path / log_file
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file_path),
            logging.StreamHandler()
        ]
    )
    
    logger.info(f"Logging to {log_file_path}")


def format_time(seconds: float) -> str:
    """
    Format seconds into human-readable time.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_lr(optimizer: torch.optim.Optimizer) -> float:
    """
    Get current learning rate from optimizer.
    
    Args:
        optimizer: PyTorch optimizer
        
    Returns:
        Current learning rate
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']
    return 0.0
