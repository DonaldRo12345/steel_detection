@echo off
REM run_after_detr.bat
REM Run from steel_defect_project directory
REM Evaluates DETR-Lite then trains Enhanced YOLOv8

echo === Step 1: Evaluate DETR-Lite ===
python tests\eval_detr.py ^
    --model results\models\detr_best.pth ^
    --splits data\processed\splits.json ^
    --annotations data\raw\NEU-DET\ANNOTATIONS ^
    --out results\metrics\eval_detr_real.json ^
    --conf-thresh 0.25

if %errorlevel% neq 0 (
    echo DETR evaluation failed!
    exit /b 1
)
echo DETR evaluation done.

echo === Step 2: Train Enhanced YOLOv8 (30 epochs on CLAHE dataset) ===
python src\train_yolo_enhanced.py ^
    --data data\processed\yolo_clahe\data.yaml ^
    --model results\models\yolo_best.pt ^
    --epochs 30 ^
    --batch-size 8 ^
    --img-size 640 ^
    --workers 0 ^
    --seed 42 ^
    --name yolo_clahe_30ep ^
    --output-dir results\models ^
    --no-copypaste ^
    --no-mixup ^
    --class-weights results\metrics\eval_yolo_perclass.json ^
    2>&1

echo === All done ===
