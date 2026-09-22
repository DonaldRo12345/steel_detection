#!/bin/bash
# Etape 0 : preparation du conteneur (a lancer UNE FOIS dans le terminal du conteneur)
# Ensuite : sauvegarder manuellement le conteneur en image via "Gestion des images"
# (la sauvegarde automatique a la fermeture du conteneur n'est pas fiable).

set -e

PROJECT_DIR="${PROJECT_DIR:-$HOME/PROJECT/steel_detection/steel_defect_project}"

echo "=========================================="
echo "Setup environnement serveur - Steel Defect"
echo "=========================================="

echo "Step 1: GPU / CUDA"
nvidia-smi || echo "Attention: pas de GPU visible dans ce conteneur"

echo ""
echo "Step 2: Python"
python --version
pip --version

echo ""
echo "Step 3: Dependances projet"
cd "$PROJECT_DIR"
# torch/torchvision sont deja fournis par l'image CUDA du serveur : on ne les reinstalle pas
pip install --no-cache-dir $(grep -vE '^\s*#|^\s*$|^torch|^torchvision' requirements.txt | tr '\n' ' ')

echo ""
echo "Step 4: Verification torch + CUDA"
python -c "import torch, ultralytics; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'); print('ultralytics', ultralytics.__version__)"

echo ""
echo "OK. Sauvegardez maintenant ce conteneur en image (ex: steel-defect:cu118-ultralytics)"
