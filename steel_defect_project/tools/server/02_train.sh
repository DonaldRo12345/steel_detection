#!/bin/bash
# Etape 2 : lancer un entrainement en arriere-plan (survit a la fermeture de
# l'onglet Jupyter / du terminal web). Logs dans experiments/logs/.
#
# Exemples :
#   ./tools/server/02_train.sh yolo_enhanced --epochs 150 --batch-size 32 --clahe
#   ./tools/server/02_train.sh yolo          --epochs 100 --batch-size 32
#   ./tools/server/02_train.sh rtdetr        --epochs 100 --batch-size 8
#   ./tools/server/02_train.sh yolo11        --epochs 150 --batch-size 32

set -e

PROJECT_DIR="${PROJECT_DIR:-$HOME/PROJECT/steel_detection/steel_defect_project}"
cd "$PROJECT_DIR"

VARIANT="${1:-yolo_enhanced}"; shift || true
DATA_YOLO="${DATA:-data/processed/yolo/data.yaml}"
STAMP=$(date +%Y%m%d_%H%M%S)
RUN_NAME="${VARIANT}_${STAMP}"
mkdir -p experiments/logs
LOG="experiments/logs/${RUN_NAME}.log"
export PYTHONPATH="$PROJECT_DIR/src:$PYTHONPATH"

case "$VARIANT" in
    yolo|yolo11|yolo_enhanced|rtdetr)
        case "$VARIANT" in
            yolo)          SCRIPT=src/train_yolo.py ;;
            yolo11)        SCRIPT=src/train_yolo11.py ;;
            yolo_enhanced) SCRIPT=src/train_yolo_enhanced.py ;;
            rtdetr)        SCRIPT=src/train_rtdetr.py ;;
        esac
        ARGS=(--data "$DATA_YOLO" --name "$RUN_NAME" --output-dir results/models)
        ;;
    detr)
        SCRIPT=src/train_detr.py
        ARGS=(--data data/processed/coco --output-dir results/models)
        ;;
    fasterrcnn)
        SCRIPT=src/train_fasterrcnn.py
        ARGS=(--data data/processed/splits.json --annotations data/raw/NEU-DET/ANNOTATIONS --output-dir results/models)
        ;;
    *) echo "Variante inconnue: $VARIANT (yolo|yolo11|yolo_enhanced|rtdetr|detr|fasterrcnn)"; exit 1 ;;
esac

echo "Script : $SCRIPT"
echo "Log    : $LOG"
echo "GPU    : ${CUDA_VISIBLE_DEVICES:-toutes}"

nohup python "$SCRIPT" "${ARGS[@]}" "$@" > "$LOG" 2>&1 &

echo "PID    : $!"
echo ""
echo "Suivi  : tail -f $LOG"
echo "GPU    : watch -n 5 nvidia-smi"
