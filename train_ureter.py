from ultralytics import YOLO
from pathlib import Path
import os

REPO_ROOT = Path(__file__).resolve().parent
DATA_YAML = Path(
    os.environ.get(
        "DATA_YAML",
        REPO_ROOT / "dataset" / "ureter_labelled_training_0609_resized_720_yolo" / "data.yaml",
    )
)
BASE_WEIGHTS = os.environ.get("YOLO_BASE_WEIGHTS", "yolo11x.pt")

model = YOLO(BASE_WEIGHTS)

results = model.train(
    data=str(DATA_YAML),
    epochs=150,
    imgsz=720,
    batch=32,
    device=[2],
    project=str(Path(__file__).parent / "runs" / "detect"),
    name="ureter_train_0609_nopadding",
)
