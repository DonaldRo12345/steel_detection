# Steel Surface Defect Detection - Docker Container
# This Dockerfile provides a reproducible environment for training and inference

FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TORCH_HOME=/workspace/.cache/torch

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3-dev \
    git \
    wget \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic links for python
RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Set working directory
WORKDIR /workspace

# Copy requirements file
COPY requirements.txt /workspace/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . /workspace/

# Create necessary directories
RUN mkdir -p /workspace/data/raw /workspace/data/processed \
    /workspace/results/models /workspace/results/metrics /workspace/results/visuals \
    /workspace/experiments/logs

# Set Python path
ENV PYTHONPATH=/workspace/src:$PYTHONPATH

# Expose Jupyter port (optional)
EXPOSE 8888

# Default command (can be overridden)
CMD ["bash"]

# Usage examples:
# Build: docker build -t steel-defect-detection .
# Run with GPU: docker run --gpus all -it -v $(pwd)/data:/workspace/data steel-defect-detection
# Run Jupyter: docker run --gpus all -p 8888:8888 -v $(pwd):/workspace steel-defect-detection jupyter notebook --ip=0.0.0.0 --allow-root
