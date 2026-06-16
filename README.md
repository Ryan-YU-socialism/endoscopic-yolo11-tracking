# Endoscopic YOLO11 Tracking

YOLO11-based detection and temporal tracking pipeline for endoscopic video analysis. The project contains training scripts, dataset preparation utilities, Kalman-smoothed inference, and optional YOLO + OSTrack fusion for more stable object tracking in ERCP, ureter, and esophagus videos.

This repository is prepared for public portfolio use. Datasets, trained checkpoints, prediction outputs, and videos are intentionally excluded.

## Highlights

- Fine-tunes YOLO11 models for multiple endoscopic targets and procedures.
- Adds video-level tracking with Kalman filtering and EMA smoothing to reduce frame-to-frame jitter.
- Provides an optional YOLO + OSTrack fusion pipeline for long video sequences and drift correction.
- Includes dataset QA tools for augmentation, label visualization, and YOLO-to-LabelMe conversion.
- Keeps all data paths configurable through environment variables and command-line arguments.

## Repository Layout

```text
.
├── train_ercp.py                 # YOLO11 training entrypoint for ERCP
├── train_ureter.py               # YOLO11 training entrypoint for ureter videos
├── train_esophagus.py            # YOLO11 training entrypoint for esophagus videos
├── predict_ercp.py               # Image-folder prediction helper
├── predict_ureter.py             # Image-folder prediction helper
├── detect_kalman_*.py            # YOLO detection + Kalman/EMA video smoothing
├── detect_track_*.py             # YOLO + OSTrack + Kalman fusion pipelines
├── yolo_to_labelme_json.py       # Convert YOLO prediction txt files to LabelMe JSON
└── dataset/
    ├── augment_*.py              # Dataset augmentation utilities
    ├── visualize_labels.py       # Render YOLO labels for QA
    └── */data.yaml, classes.txt  # Dataset metadata only; images/labels are excluded
```

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For GPU training, install the PyTorch build that matches your CUDA version before installing the remaining dependencies.

## Data And Weights

The repository does not include datasets, trained weights, run outputs, or videos.

Expected local folders:

```text
weights/
├── ercp_best.pt
├── ureter_best.pt
├── esophagus_best.pt
└── yolo11x.pt

data/
├── images/
│   ├── ercp_val/
│   └── ureter_val/
└── videos/
    ├── ercp/
    ├── ureter/
    └── esophagus/
```

Download options:

- YOLO11 base weights: download from the official Ultralytics model release or let `ultralytics.YOLO("yolo11x.pt")` fetch the model automatically.
- Fine-tuned `*_best.pt`: train with the scripts in this repository or place externally trained checkpoints in `weights/`.
- OSTrack: clone/install the official OSTrack project under `external/OSTrack/` and place its checkpoint at `external/OSTrack/checkpoints/pytorch_model.bin` when using `detect_track_*.py`.

## Training

Each training script uses the dataset metadata under `dataset/` by default. Put the actual `images/` and `labels/` folders under the selected dataset directory locally.

```bash
python train_ercp.py
python train_ureter.py
python train_esophagus.py
```

Override the dataset or base model when needed:

```bash
DATA_YAML=/path/to/data.yaml YOLO_BASE_WEIGHTS=/path/to/yolo11x.pt python train_ercp.py
```

## Prediction

```bash
YOLO_WEIGHTS=weights/ercp_best.pt TEST_DIR=data/images/ercp_val python predict_ercp.py
YOLO_WEIGHTS=weights/ureter_best.pt TEST_DIR=data/images/ureter_val python predict_ureter.py
```

## Video Tracking

Kalman-smoothed YOLO tracking:

```bash
YOLO_WEIGHTS=weights/ercp_best.pt VIDEO_DIR=data/videos/ercp python detect_kalman_ercp.py
YOLO_WEIGHTS=weights/ureter_best.pt VIDEO_DIR=data/videos/ureter python detect_kalman_ureter.py
YOLO_WEIGHTS=weights/esophagus_best.pt VIDEO_DIR=data/videos/esophagus python detect_kalman_esophagus.py
```

YOLO + OSTrack fusion:

```bash
OSTRACK_ROOT=external/OSTrack \
OSTRACK_CKPT=external/OSTrack/checkpoints/pytorch_model.bin \
YOLO_WEIGHTS=weights/ercp_best.pt \
VIDEO_DIR=data/videos/ercp \
python detect_track_ercp.py
```

## Resume-Friendly Summary

Built an endoscopic video perception pipeline using YOLO11 fine-tuning and Kalman/OSTrack temporal fusion, including dataset augmentation, label QA, multi-GPU batch inference, and reproducible training/prediction scripts without exposing private medical datasets or model checkpoints.
