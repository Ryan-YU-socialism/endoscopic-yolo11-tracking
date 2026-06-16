import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def yolo_line_to_xyxy_conf(
    line: str, img_w: int, img_h: int, has_conf: bool = True
) -> Tuple[int, float, float, float, float, Optional[float]]:
    """Convert one normalized YOLO label line to pixel-space xyxy coordinates."""
    parts = line.strip().split()
    if not parts:
        raise ValueError("Empty line")

    if has_conf:
        if len(parts) < 6:
            raise ValueError(f"Expect 6 fields (cls cx cy w h conf), got {len(parts)}: {line}")
        cls = int(float(parts[0]))
        cx, cy, bw, bh = map(float, parts[1:5])
        conf = float(parts[5])
    else:
        if len(parts) < 5:
            raise ValueError(f"Expect 5 fields (cls cx cy w h), got {len(parts)}: {line}")
        cls = int(float(parts[0]))
        cx, cy, bw, bh = map(float, parts[1:5])
        conf = None

    x_c = cx * img_w
    y_c = cy * img_h
    box_w = bw * img_w
    box_h = bh * img_h

    x1 = max(0.0, min(float(img_w - 1), x_c - box_w / 2.0))
    y1 = max(0.0, min(float(img_h - 1), y_c - box_h / 2.0))
    x2 = max(0.0, min(float(img_w - 1), x_c + box_w / 2.0))
    y2 = max(0.0, min(float(img_h - 1), y_c + box_h / 2.0))

    return cls, x1, y1, x2, y2, conf


def xyxy_to_labelme_rect_points(x1: float, y1: float, x2: float, y2: float) -> List[List[float]]:
    """LabelMe rectangles use two points: top-left and bottom-right."""
    return [[float(x1), float(y1)], [float(x2), float(y2)]]


def build_labelme_json(
    image_path_name: str,
    img_w: int,
    img_h: int,
    shapes: List[dict],
    version: str = "2.3.6",
) -> dict:
    return {
        "version": version,
        "flags": {},
        "shapes": shapes,
        "imagePath": image_path_name,
        "imageData": None,
        "imageHeight": int(img_h),
        "imageWidth": int(img_w),
        "text": "",
    }


def convert_one_image(
    img_path: Path,
    label_path: Path,
    out_json_path: Path,
    class_map: Dict[int, str],
    has_conf: bool = True,
    keep_top1: bool = False,
    min_conf: float = 0.0,
) -> None:
    img = cv2.imread(str(img_path))
    if img is None:
        raise RuntimeError(f"Failed to read image: {img_path}")
    img_h, img_w = img.shape[:2]

    shapes = []
    if label_path.exists():
        lines = label_path.read_text(encoding="utf-8").strip().splitlines()
        dets = []
        for line in lines:
            if not line.strip():
                continue
            cls, x1, y1, x2, y2, conf = yolo_line_to_xyxy_conf(line, img_w, img_h, has_conf=has_conf)
            if conf is not None and conf < min_conf:
                continue
            dets.append((cls, x1, y1, x2, y2, conf))

        if keep_top1 and dets:
            dets.sort(key=lambda item: (item[5] if item[5] is not None else -1.0), reverse=True)
            dets = dets[:1]

        for cls, x1, y1, x2, y2, conf in dets:
            shapes.append(
                {
                    "label": class_map.get(cls, str(cls)),
                    "points": xyxy_to_labelme_rect_points(x1, y1, x2, y2),
                    "group_id": None,
                    "description": (f"conf: {conf:.2f}" if conf is not None else ""),
                    "difficult": False,
                    "shape_type": "rectangle",
                    "flags": {},
                    "attributes": {},
                }
            )

    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    data = build_labelme_json(img_path.name, img_w, img_h, shapes)
    out_json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_class_map(values: List[str]) -> Dict[int, str]:
    class_map = {0: "Entry", 1: "Lumen", 2: "Calculus"}
    for value in values:
        cls_id, label = value.split(":", 1)
        class_map[int(cls_id)] = label
    return class_map


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Ultralytics YOLO txt predictions to LabelMe JSON files.")
    parser.add_argument(
        "--pred-root",
        type=Path,
        default=Path("outputs/detect/test_results"),
        help="YOLO predict output root. Images and labels/ directories are searched recursively.",
    )
    parser.add_argument(
        "--json-out-root",
        type=Path,
        default=Path("outputs/detect/test_results_json"),
        help="Directory where LabelMe JSON files will be written.",
    )
    parser.add_argument("--class-map", action="append", default=[], help="Class mapping like 0:Entry. May repeat.")
    parser.add_argument("--no-conf", action="store_true", help="Set when YOLO txt files do not include confidence.")
    parser.add_argument("--keep-top1", action="store_true", help="Keep only the highest-confidence box per image.")
    parser.add_argument("--min-conf", type=float, default=0.0, help="Filter boxes below this confidence.")
    args = parser.parse_args()

    class_map = parse_class_map(args.class_map)
    img_paths = sorted(p for p in args.pred_root.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS)

    print(f"[INFO] Found {len(img_paths)} images under {args.pred_root}")

    for img_path in img_paths:
        label_path = img_path.parent / "labels" / (img_path.stem + ".txt")
        rel_dir = img_path.parent.relative_to(args.pred_root)
        out_json_path = args.json_out_root / rel_dir / (img_path.stem + ".json")

        try:
            convert_one_image(
                img_path=img_path,
                label_path=label_path,
                out_json_path=out_json_path,
                class_map=class_map,
                has_conf=not args.no_conf,
                keep_top1=args.keep_top1,
                min_conf=args.min_conf,
            )
        except Exception as exc:
            print(f"[WARN] Failed: {img_path} | {exc}")

    print(f"[DONE] JSON saved to: {args.json_out_root}")


if __name__ == "__main__":
    main()
