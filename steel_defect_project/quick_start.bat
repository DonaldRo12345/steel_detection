@echo off
REM Quick Start Script for Steel Defect Detection Project
REM This script sets up the environment and runs a complete demo pipeline

echo ==========================================
echo Steel Defect Detection - Quick Start
echo ==========================================
echo.

REM Step 1: Check Python
echo Step 1: Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python not found. Please install Python 3.8+.
    exit /b 1
)
python --version
echo.

REM Step 2: Install dependencies
echo Step 2: Installing dependencies...
set /p INSTALL="Install dependencies? (y/n) "
if /i "%INSTALL%"=="y" (
    pip install -r requirements.txt
    echo ✓ Dependencies installed
)
echo.

REM Step 3: Download dataset instructions
echo Step 3: Dataset setup
if not exist "data\raw\IMAGES" (
    echo ⚠ NEU-DET dataset not found!
    echo.
    echo Please download the dataset:
    echo   1. Visit: http://faculty.neu.edu.cn/yunhyan/NEU_surface_defect_database.html
    echo   2. Download NEU-DET.zip
    echo   3. Extract to: data\raw\
    echo   4. Verify structure: data\raw\IMAGES\ and data\raw\ANNOTATIONS\
    echo.
    pause
)
echo.

REM Step 4: Prepare dataset
echo Step 4: Preparing dataset...
if not exist "data\processed\yolo" (
    echo Converting dataset to YOLO and COCO formats...
    python src\data_utils.py --action prepare --input data\raw --output data\processed --formats yolo,coco --seed 42
    echo ✓ Dataset prepared
) else (
    echo ✓ Dataset already prepared
)
echo.

REM Step 5: Visualize samples
echo Step 5: Generating sample visualizations...
python src\data_utils.py --action visualize --input data\processed\yolo --output results\visuals --num-samples 10 --seed 42
echo ✓ Visualizations saved to results\visuals\
echo.

REM Step 6: Quick training demo
echo Step 6: Running quick training demo ^(1 epoch^)...
set /p TRAIN="Run training demo? This will take a few minutes. (y/n) "
if /i "%TRAIN%"=="y" (
    echo Training YOLO ^(1 epoch^)...
    python src\train_yolo.py --data data\processed\yolo\data.yaml --epochs 1 --batch-size 4 --device cpu --name yolo_demo
    echo ✓ Demo training complete
)
echo.

REM Step 7: Instructions for full training
echo ==========================================
echo Quick Start Complete!
echo ==========================================
echo.
echo Next steps:
echo.
echo 1. Full YOLO training ^(50 epochs, GPU recommended^):
echo    python src\train_yolo.py --data data\processed\yolo\data.yaml --epochs 50 --device cuda:0
echo.
echo 2. Full DETR training ^(100 epochs, GPU recommended^):
echo    python src\train_detr.py --data data\processed\coco --epochs 100 --device cuda:0
echo.
echo 3. Evaluate trained model:
echo    python src\eval.py --model results\models\yolo_best.pt --data data\processed\yolo\data.yaml
echo.
echo 4. Run inference:
echo    python src\infer.py --model results\models\yolo_best.pt --image tests\sample.jpg
echo.
echo 5. Explore Jupyter demo:
echo    jupyter notebook notebooks\demo_inference.ipynb
echo.
echo For more details, see README.md
echo ==========================================
pause
