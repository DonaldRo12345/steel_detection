#!/bin/bash
# Etape 1 : brancher le dataset partage du serveur sur le projet, puis regenerer
# les splits YOLO/COCO (data.yaml contient un chemin ABSOLU, il doit etre
# regenere sur le serveur : celui du depot pointe vers un chemin Windows).

set -e

DATASET_DIR="${DATASET_DIR:-$HOME/DATASET/NEU-DET}"     # dataset public partage
PROJECT_DIR="${PROJECT_DIR:-$HOME/PROJECT/steel_detection/steel_defect_project}"

echo "Dataset partage : $DATASET_DIR"
echo "Projet          : $PROJECT_DIR"

if [ ! -d "$DATASET_DIR/IMAGES" ]; then
    echo "Erreur: $DATASET_DIR/IMAGES introuvable."
    echo "Deposez NEU-DET dans DATASET/NEU-DET (IMAGES/ + ANNOTATIONS/) via 'Mes fichiers'."
    exit 1
fi

cd "$PROJECT_DIR"
mkdir -p data/raw

# Lien symbolique : les donnees ne sont PAS copiees dans le projet (dataset public partage)
if [ ! -e data/raw/NEU-DET ]; then
    ln -s "$DATASET_DIR" data/raw/NEU-DET
    echo "Lien cree : data/raw/NEU-DET -> $DATASET_DIR"
fi

echo ""
echo "Generation des splits YOLO + COCO..."
python src/data_utils.py --action prepare --input data/raw/NEU-DET --output data/processed --formats yolo,coco

echo ""
echo "data.yaml genere :"
cat data/processed/yolo/data.yaml
