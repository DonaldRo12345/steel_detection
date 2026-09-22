#!/bin/bash
# Etape 1 : brancher le dataset partage du serveur sur le projet, puis regenerer
# les splits YOLO/COCO (data.yaml contient un chemin ABSOLU, il doit etre
# regenere sur le serveur : celui du depot pointe vers un chemin Windows).

set -e

# Racine du stockage "Mes fichiers" : $HOME si le conteneur tourne sous l'utilisateur,
# /home/user si le conteneur tourne en root (le stockage reste monte sous /home/user).
if [ -z "$DATASET_DIR" ]; then
    for candidat in "$HOME/DATASET/NEU-DET" "/home/user/DATASET/NEU-DET"; do
        if [ -d "$candidat/IMAGES" ]; then DATASET_DIR="$candidat"; break; fi
    done
    DATASET_DIR="${DATASET_DIR:-$HOME/DATASET/NEU-DET}"   # defaut si rien trouve : message d'erreur explicite
fi
PROJECT_DIR="${PROJECT_DIR:-$HOME/PROJECT/steel_detection/steel_defect_project}"

echo "Dataset partage : $DATASET_DIR"
echo "Projet          : $PROJECT_DIR"

if [ ! -d "$DATASET_DIR/IMAGES" ]; then
    echo "Erreur: $DATASET_DIR/IMAGES introuvable."
    echo "Deposez NEU-DET dans DATASET/NEU-DET (IMAGES/ + ANNOTATIONS/) via 'Mes fichiers'."
    echo "Si le dataset est ailleurs : export DATASET_DIR=/chemin/vers/NEU-DET puis relancer."
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
