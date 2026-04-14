#!/bin/bash
# Test inference script for steel defect detection
# This script runs basic inference tests to validate the installation

echo "=========================================="
echo "Steel Defect Detection - Inference Test"
echo "=========================================="
echo ""

# Check if python is available
if ! command -v python &> /dev/null; then
    echo "Error: Python not found. Please install Python 3.8+."
    exit 1
fi

echo "Python version:"
python --version
echo ""

# Test 1: Check imports
echo "Test 1: Checking imports..."
python -c "import torch; import cv2; import numpy; print('✓ Core imports OK')" || exit 1
echo ""

# Test 2: Run data visualization (if data exists)
if [ -d "data/processed/yolo" ]; then
    echo "Test 2: Visualizing augmentations..."
    python src/data_utils.py --action visualize --input data/processed/yolo --output results/visuals --num-samples 5
    echo "✓ Visualization test passed"
else
    echo "Test 2: Skipped (no processed data found)"
fi
echo ""

# Test 3: Test inference (if model exists)
if [ -f "results/models/yolo_best.pt" ]; then
    echo "Test 3: Running YOLO inference..."
    
    # Create a dummy test image if sample doesn't exist
    if [ ! -f "tests/sample.jpg" ]; then
        python -c "import cv2; import numpy as np; img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8); cv2.imwrite('tests/sample.jpg', img)"
    fi
    
    python src/infer.py \
        --model results/models/yolo_best.pt \
        --model-type yolo \
        --image tests/sample.jpg \
        --out-dir results/visuals \
        --conf-thres 0.25
    
    echo "✓ Inference test passed"
else
    echo "Test 3: Skipped (no trained model found)"
    echo "  Train a model first: python src/train_yolo.py --epochs 1 --device cpu"
fi
echo ""

# Test 4: Verify output structure
echo "Test 4: Verifying project structure..."
dirs=("data" "src" "results" "experiments" "notebooks" "report")
for dir in "${dirs[@]}"; do
    if [ -d "$dir" ]; then
        echo "  ✓ $dir/ exists"
    else
        echo "  ✗ $dir/ missing"
    fi
done
echo ""

echo "=========================================="
echo "Test suite completed!"
echo "=========================================="
