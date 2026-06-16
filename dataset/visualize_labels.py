import argparse
import os
from pathlib import Path

import cv2
import numpy as np

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def load_classes(classes_path: Path) -> list[str]:
    if not classes_path.exists():
        return []
    return [item.strip() for item in classes_path.read_text(encoding="utf-8").splitlines() if item.strip()]


def yolo_to_xyxy(cx: float, cy: float, w: float, h: float, img_w: int, img_h: int) -> tuple[int, int, int, int]:
    x1 = (cx - w / 2.0) * img_w
    y1 = (cy - h / 2.0) * img_h
    x2 = (cx + w / 2.0) * img_w
    y2 = (cy + h / 2.0) * img_h
    return int(x1), int(y1), int(x2), int(y2)


def visualize_image(img_path: Path, label_path: Path, classes: list[str]) -> np.ndarray:
    image = cv2.imread(str(img_path))
    if image is None:
        raise RuntimeError(f"Failed to read image: {img_path}")

    height, width = image.shape[:2]
    if not label_path.exists():
        return image

    for line in label_path.read_text(encoding="utf-8").strip().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls = int(parts[0])
        cx, cy, bw, bh = map(float, parts[1:5])
        x1, y1, x2, y2 = yolo_to_xyxy(cx, cy, bw, bh, width, height)
        color = (0, 255, 0)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        label = classes[cls] if cls < len(classes) else str(cls)
        cv2.putText(image, label, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return image


def gather_image_paths(image_root: Path) -> list[Path]:
    if not image_root.exists():
        return []

    if any((image_root / split).exists() for split in ("train", "val", "test")):
        image_paths = []
        for split in ("train", "val", "test"):
            split_dir = image_root / split
            if split_dir.exists():
                image_paths.extend(sorted(path for path in split_dir.iterdir() if path.suffix.lower() in IMG_EXTS))
        return image_paths

    return sorted(path for path in image_root.iterdir() if path.suffix.lower() in IMG_EXTS)


def process_dataset(data_root: Path, out_root: Path | None, limit: int | None) -> None:
    image_root = data_root / "images"
    label_root = data_root / "labels"
    classes = load_classes(data_root / "classes.txt")

    out_root = out_root or (data_root / "visualization_first100")
    out_root.mkdir(parents=True, exist_ok=True)

    image_paths = gather_image_paths(image_root)
    if limit is not None:
        image_paths = image_paths[:limit]

    print(f"Dataset: {data_root} - processing {len(image_paths)} images")

    for img_path in image_paths:
        rel = img_path.relative_to(image_root)
        split_label_dir = label_root / rel.parent
        label_path = split_label_dir / f"{img_path.stem}.txt" if split_label_dir.exists() else label_root / f"{img_path.stem}.txt"

        visualization = visualize_image(img_path, label_path, classes)
        out_path = out_root / rel.parent / img_path.name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), visualization)

    print(f"Saved visualizations to: {out_root}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render YOLO labels on top of dataset images for quick QA.")
    parser.add_argument("--data-root", type=Path, default=None, help="Dataset root containing images/, labels/, classes.txt.")
    parser.add_argument("--out", type=Path, default=None, help="Output visualization directory.")
    parser.add_argument("--limit", type=int, default=100, help="Maximum images to process.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    default_data_root = Path(
        os.environ.get("DATA_ROOT", repo_root / "dataset" / "ureter_labelled_0609_resized_yolo_nopadding")
    )

    process_dataset(args.data_root or default_data_root, args.out, args.limit)


if __name__ == "__main__":
    main()
