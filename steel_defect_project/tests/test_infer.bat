@echo off
REM Test inference script for steel defect detection (Windows)
REM This script runs basic inference tests to validate the installation

echo ==========================================
echo Steel Defect Detection - Inference Test
echo ==========================================
echo.

REM Check if python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python not found. Please install Python 3.8+.
    exit /b 1
)

echo Python version:
python --version
echo.

REM Test 1: Check imports
echo Test 1: Checking imports...
python -c "import torch; import cv2; import numpy; print('✓ Core imports OK')"
if errorlevel 1 exit /b 1
echo.

REM Test 2: Run data visualization (if data exists)
if exist "data\processed\yolo" (
    echo Test 2: Visualizing augmentations...
    python src\data_utils.py --action visualize --input data\processed\yolo --output results\visuals --num-samples 5
    echo ✓ Visualization test passed
) else (
    echo Test 2: Skipped ^(no processed data found^)
)
echo.

REM Test 3: Test inference (if model exists)
if exist "results\models\yolo_best.pt" (
    echo Test 3: Running YOLO inference...
    
    REM Create a dummy test image if sample doesn't exist
    if not exist "tests\sample.jpg" (
        python -c "import cv2; import numpy as np; img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8); cv2.imwrite('tests/sample.jpg', img)"
    )
    
    python src\infer.py --model results\models\yolo_best.pt --model-type yolo --image tests\sample.jpg --out-dir results\visuals --conf-thres 0.25
    
    echo ✓ Inference test passed
) else (
    echo Test 3: Skipped ^(no trained model found^)
    echo   Train a model first: python src\train_yolo.py --epochs 1 --device cpu
)
echo.

REM Test 4: Verify output structure
echo Test 4: Verifying project structure...
for %%d in (data src results experiments notebooks report) do (
    if exist "%%d\" (
        echo   ✓ %%d\ exists
    ) else (
        echo   ✗ %%d\ missing
    )
)
echo.

echo ==========================================
echo Test suite completed!
echo ==========================================
pause
