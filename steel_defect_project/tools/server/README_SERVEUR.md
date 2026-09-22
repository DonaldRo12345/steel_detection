# Intégration du serveur GPU de l'école au projet

Serveur : plateforme conteneurisée (Mes fichiers / Environnement expérimental / Gestion des images).
Doc plateforme : http://192.168.249.200:8008/docsify/#/teach/user/

## 1. Où mettre quoi (conventions du serveur)

| Contenu | Emplacement serveur | Pourquoi |
|---|---|---|
| NEU-DET (dataset public) | `DATASET/NEU-DET/{IMAGES,ANNOTATIONS}` | dataset public réutilisable par plusieurs projets |
| Code du projet | `PROJECT/steel_detection/` (clone git) | code versionné, propre au projet |
| Splits générés, poids, runs | dans le projet (`data/processed/`, `results/`, `runs/`) | régénérables, non versionnés |
| Scripts réutilisables | `utils/` | scripts bash/python communs |
| Reste | `杂项` (divers) | garder `home` propre |

Le dépôt git ne contient **pas** les données ni les runs (voir `.gitignore`) : le dataset
transite par « Mes fichiers », le code par git.

## 2. Mise en place (une seule fois)

1. **Dataset** — uploader `NEU-DET.zip` dans `DATASET/`, dézipper pour obtenir
   `DATASET/NEU-DET/IMAGES` et `DATASET/NEU-DET/ANNOTATIONS`.
2. **Projet** — créer un projet dans « Environnement expérimental > développement
   tout-en-un », y créer un conteneur GPU depuis une image CUDA + PyTorch, puis
   ouvrir un **terminal** :
   ```bash
   mkdir -p ~/PROJECT && cd ~/PROJECT
   git clone <url-du-depot> steel_detection
   cd steel_detection/steel_defect_project
   ```
3. **Environnement** :
   ```bash
   ./tools/server/00_setup_env.sh
   ```
   (installe les dépendances sans réinstaller torch/torchvision fournis par l'image,
   puis vérifie `torch.cuda.is_available()`).
4. **Sauvegarder l'image manuellement** dans « Gestion des images »
   (ex. `steel-defect:cu118-ultralytics`) — la sauvegarde automatique à la fermeture
   du conteneur n'est pas fiable. Les conteneurs suivants partiront de cette image
   et n'auront plus besoin de l'étape 3.

## 3. Données

```bash
./tools/server/01_prepare_data.sh
```
- crée `data/raw/NEU-DET` → lien symbolique vers `~/DATASET/NEU-DET` (pas de copie) ;
- régénère `data/processed/yolo|coco` et surtout **`data.yaml`**, dont le champ `path`
  est un chemin **absolu** (celui du dépôt pointe vers `C:\Users\...` : inutilisable
  sous Linux). Ne pas commiter le `data.yaml` du serveur.

Variables d'environnement si vos chemins diffèrent :
```bash
export DATASET_DIR=$HOME/DATASET/NEU-DET
export PROJECT_DIR=$HOME/PROJECT/steel_detection/steel_defect_project
```

## 4. Entraînement

```bash
./tools/server/02_train.sh yolo_enhanced --epochs 150 --batch-size 32 --clahe
tail -f experiments/logs/yolo_enhanced_<date>.log
watch -n 5 nvidia-smi
```
Le script lance en `nohup` : l'entraînement survit à la fermeture de l'onglet
Jupyter/terminal web (une session Jupyter fermée tue sinon le processus).
Variantes : `yolo`, `yolo11`, `yolo_enhanced`, `rtdetr`, `detr`, `fasterrcnn`.

Sur GPU, augmenter par rapport aux valeurs locales (réglées pour du CPU) :
`--batch-size 32` (voire 64 selon la VRAM), `--workers 8`, `--epochs 100-200`.

## 5. Récupérer les résultats

Le poids final est le seul artefact à ramener en local / dans git :
```bash
# depuis le conteneur, copier vers un dossier visible dans « Mes fichiers »
cp results/models/<run>/weights/best.pt ~/PROJECT/steel_detection/best_<run>.pt
```
puis télécharger via « Mes fichiers » et le placer en local dans
`results/models/yolo_enhanced_best.pt` (nom attendu par `app.py` / `Dockerfile.web`).
Les dossiers `runs/` (134 Mo) et `results/` (1,1 Go) restent sur le serveur.

## 6. Points de vigilance

- `data.yaml` : chemin absolu → toujours régénéré par l'étape 3, jamais commité depuis le serveur.
- `src/train_yolo11.py` : le device était codé en dur sur `cpu` et les hyperparamètres
  (50 époques, batch 8) n'étaient pas pilotables ; le script a désormais les mêmes options
  que les autres (`--epochs`, `--batch-size`, `--device`, `--name`...) et auto-détecte le GPU.
- Les images pré-entraînées (`yolo11n.pt`, `yolov8n.pt`, `rtdetr-l.pt`) sont téléchargées
  par ultralytics au premier run : si le conteneur n'a pas Internet, les copier depuis
  le poste local dans le dossier du projet.
- Toujours travailler dans un conteneur sauvegardé en image après changement d'environnement.
