from ultralytics import YOLO
from pathlib import Path
import os

REPO_ROOT = Path(__file__).resolve().parent
WEIGHTS = os.environ.get("YOLO_WEIGHTS", str(REPO_ROOT / "weights" / "ercp_best.pt"))
TEST_DIR = os.environ.get("TEST_DIR", str(REPO_ROOT / "data" / "images" / "ercp_val"))
OUT_DIR = os.environ.get("OUT_DIR", str(REPO_ROOT / "outputs" / "detect" / "ercp_predict"))

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
CONF = 0.25
IOU = 0.5
IMGSZ = 480
DEVICE = 0


def main():
    model = YOLO(WEIGHTS)

    test_root = Path(TEST_DIR)
    images = sorted([p for p in test_root.iterdir() if p.suffix.lower() in IMG_EXTS])
    print(f"[INFO] Found {len(images)} images in {TEST_DIR}")

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
