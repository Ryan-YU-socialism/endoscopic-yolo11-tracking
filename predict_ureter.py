from ultralytics import YOLO
from pathlib import Path
import os

REPO_ROOT = Path(__file__).resolve().parent
WEIGHTS = os.environ.get("YOLO_WEIGHTS", str(REPO_ROOT / "weights" / "ureter_best.pt"))
TEST_DIR = os.environ.get("TEST_DIR", str(REPO_ROOT / "data" / "images" / "ureter_val"))
OUT_DIR = os.environ.get("OUT_DIR", str(REPO_ROOT / "outputs" / "detect" / "ureter_predict"))

CONF = 0.25
IOU = 0.5
IMGSZ = 480
DEVICE = 0


def main():
    model = YOLO(WEIGHTS)

    test_root = Path(TEST_DIR)
    print(f"[INFO] Predicting on: {TEST_DIR}")

    model.predict(
        source=str(test_root),
        conf=CONF,
        iou=IOU,
        imgsz=IMGSZ,
        device=DEVICE,
        half=True,
        save=True,
        save_txt=True,
        save_conf=True,
        project=str(Path(OUT_DIR).parent),
        name=Path(OUT_DIR).name,
        exist_ok=True,
    )

    print(f"[DONE] Results saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
