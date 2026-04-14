# run_after_detr.ps1
# Waits for DETR training process to finish, then:
#   1. Runs DETR evaluation
#   2. Launches Enhanced YOLOv8 training
# Usage: powershell -File tests/run_after_detr.ps1

$proj = "c:\Users\DonaldRo\Documents\GitHub\steel_detection\steel_defect_project"
Set-Location $proj

Write-Host "=== run_after_detr.ps1 ===" -ForegroundColor Cyan
Write-Host "Watching for DETR process to finish..."

# Poll until no python process has detr in its title or until detr_best.pth appears
$detrModel = "$proj\results\models\detr_best.pth"
$maxWaitHours = 6
$pollIntervalSec = 60
$elapsed = 0

while ($elapsed -lt ($maxWaitHours * 3600)) {
    $detrRunning = Get-Process python -ErrorAction SilentlyContinue | 
                   Where-Object { $_.CPU -gt 100 }
    $modelExists = Test-Path $detrModel
    
    if ($modelExists) {
        # Check if process died (no more active python with high CPU)
        if (-not $detrRunning -or $detrRunning.Count -eq 0) {
            Write-Host "DETR training appears complete (model found, no active python)." -ForegroundColor Green
            break
        }
    }
    
    $tail = Get-Content "$proj\experiments\logs\detr_real_stdout.log" -Tail 2 -ErrorAction SilentlyContinue
    $epochLine = $tail | Select-String "Epoch (\d+)/15"
    if ($epochLine -match "Epoch 15") {
        Start-Sleep -Seconds 120  # Wait for final save
        Write-Host "Epoch 15 detected — DETR training complete." -ForegroundColor Green
        break
    }
    
    Write-Host "[$(Get-Date -Format 'HH:mm:ss')] Waiting... ($([int]($elapsed/60)) min elapsed)"
    Start-Sleep -Seconds $pollIntervalSec
    $elapsed += $pollIntervalSec
}

# ── Step 1: Evaluate DETR ─────────────────────────────────────────────────
Write-Host "`n=== Step 1: Evaluating DETR-Lite ===" -ForegroundColor Yellow
python "$proj\tests\eval_detr.py" `
    --model "$proj\results\models\detr_best.pth" `
    --splits "$proj\data\processed\splits.json" `
    --annotations "$proj\data\raw\NEU-DET\ANNOTATIONS" `
    --out "$proj\results\metrics\eval_detr_real.json" `
    --conf-thresh 0.25

if ($LASTEXITCODE -ne 0) {
    Write-Host "DETR evaluation failed — check the script." -ForegroundColor Red
    exit 1
}

Write-Host "DETR evaluation complete. Results at results/metrics/eval_detr_real.json" -ForegroundColor Green

# ── Step 2: Enhanced YOLOv8 training ─────────────────────────────────────
Write-Host "`n=== Step 2: Starting Enhanced YOLOv8 Training ===" -ForegroundColor Yellow
python "$proj\src\train_yolo_enhanced.py" `
    --data "$proj\data\processed\yolo_clahe\data.yaml" `
    --model "$proj\results\models\yolo_best.pt" `
    --epochs 30 `
    --batch-size 8 `
    --img-size 640 `
    --workers 0 `
    --seed 42 `
    --name yolo_clahe_30ep `
    --output-dir "$proj\results\models" `
    --no-copypaste `
    --no-mixup `
    --class-weights "$proj\results\metrics\eval_yolo_perclass.json" `
    2>&1 | Tee-Object "$proj\experiments\logs\yolo_clahe_stdout.log"

Write-Host "`n=== All done ===" -ForegroundColor Green
Write-Host "DETR results: $proj\results\metrics\eval_detr_real.json"
Write-Host "YOLOv8 Enhanced model: $proj\results\models\yolo_clahe_30ep\"
