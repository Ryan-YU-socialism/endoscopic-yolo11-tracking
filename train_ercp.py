from ultralytics import YOLO
from pathlib import Path
import os

REPO_ROOT = Path(__file__).resolve().parent
DATA_YAML = Path(
    os.environ.get(
        "DATA_YAML",
        REPO_ROOT / "dataset" / "ercp_labelled_training_0606_resized_720_yolo" / "data.yaml",
    )
)
BASE_WEIGHTS = os.environ.get("YOLO_BASE_WEIGHTS", "yolo11x.pt")

model = YOLO(BASE_WEIGHTS)

results = model.train(
    data=str(DATA_YAML),
    epochs=100,
    imgsz=720,
    batch=14,
    device=[0],
    project=str(Path(__file__).parent / "runs" / "detect"),
    name="ercp_train_720_nopadding",
)
