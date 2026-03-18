# Face Profiler

![Face Profiler Demo](demo.jpg)

GUI application for face detection and facial attribute analysis (age, gender, race) using MiVOLO v2 + FairFace + RetinaFace.

## Setup

### 1. Create a virtual environment

```shell
python -m venv venv
```

Activate it:

```shell
# Windows (PowerShell)
.\venv\Scripts\Activate.ps1

# Windows (CMD)
venv\Scripts\activate.bat

# Linux / macOS
source venv/bin/activate
```

### 2. Install dependencies

```shell
pip install -r requirements.txt
```

Or manually:

```shell
pip install deepface tf-keras pillow torch torchvision transformers accelerate gdown opencv-python
pip install timm==0.8.13.dev0
```

**Important:** MiVOLO requires `timm==0.8.13.dev0` specifically. Newer versions will not work.

### 3. Clone MiVOLO (only if `mivolo/` folder is missing)

The `mivolo/` package is included in this repo. This step is only needed if you're setting up from scratch:

```shell
git clone --depth 1 https://github.com/WildChlamydia/MiVOLO.git mivolo_repo
cp -r mivolo_repo/mivolo mivolo
```

On Windows (PowerShell):

```powershell
git clone --depth 1 https://github.com/WildChlamydia/MiVOLO.git mivolo_repo
Copy-Item -Recurse mivolo_repo\mivolo mivolo
```

### 4. GPU support (recommended)

Install PyTorch with CUDA. For CUDA 12.8:

```shell
pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

For other CUDA versions, see https://pytorch.org/get-started/locally/

Verify GPU is detected:

```shell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## Project structure

```
face-profiler/
├── images/                 # Default image folder
├── mivolo/                 # MiVOLO package (cloned from GitHub)
├── face_profiler.py        # GUI application
├── requirements.txt
└── README.md
```

## Usage

```shell
python face_profiler.py
```

- **RANDOM IMAGE** — picks a random image from the current folder and analyzes it
- **PICK IMAGE** — opens a file picker to select any image from anywhere on disk
- **CHANGE FOLDER** — switch the image folder (default is `images/`)

## Model stack

| Stage | Model | What it does | Accuracy |
|-------|-------|-------------|----------|
| Detection | RetinaFace | Finds faces, returns bounding boxes + confidence | 0.7 confidence threshold |
| Age | MiVOLO v2 | SOTA age estimation trained on large-scale age-diverse datasets including children, teens, adults, and elderly | ~3.65 MAE |
| Gender | MiVOLO v2 | Gender classification with confidence score | ~98% accuracy |
| Race | FairFace (ResNet34) | 7-class race classification trained on balanced, diverse dataset | Outputs dominant race with confidence |

### Pipeline rules

- Faces smaller than 40px are rejected (too small for reliable analysis)
- Detection confidence threshold: 0.7
- Age is shown as both a group estimate (e.g., "13-17") and exact estimate (e.g., "14.6 years")
- All predictions include confidence percentages
- Face crops are expanded by 20% before analysis for better context
- Each detected person gets a distinct color with leader lines to info cards

### Model downloads (automatic on first run)

| Model | Size | Source |
|-------|------|--------|
| MiVOLO v2 | ~300MB | HuggingFace (`iitolstykh/mivolo_v2`) |
| FairFace ResNet34 | ~85MB | Google Drive (original FairFace authors) |
| RetinaFace | ~100MB | Cached by retinaface package |

All models are cached locally after first download. No API calls during inference.

### FairFace race classes

White, Black, Latino/Hispanic, East Asian, Southeast Asian, Indian, Middle Eastern

## Notes

- First run is slow — models are downloaded and cached. Subsequent runs are fast.
- GPU (CUDA) speeds up inference substantially but CPU works fine.
- All inference runs locally on your machine.
- HiDPI / 4K displays are supported with automatic scaling.
