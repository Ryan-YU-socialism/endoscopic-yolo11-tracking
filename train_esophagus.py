from ultralytics import YOLO
from pathlib import Path
import os

REPO_ROOT = Path(__file__).resolve().parent
DATA_YAML = Path(
    os.environ.get(
        "DATA_YAML",
        REPO_ROOT / "dataset" / "esophagus_resized_yolo_nopadding" / "data.yaml",
    )
)
BASE_WEIGHTS = os.environ.get("YOLO_BASE_WEIGHTS", "yolo11x.pt")

model = YOLO(BASE_WEIGHTS)

results = model.train(
    data=str(DATA_YAML),
    epochs=100,
    imgsz=480,
    batch=69,
    device=[0,1,2],
    project=str(Path(__file__).parent / "runs" / "detect"),
    name="esophagus_train_nopadding",
)
