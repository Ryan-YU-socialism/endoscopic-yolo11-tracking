import argparse
import cv2
import numpy as np
import random
import os
from pathlib import Path

IMG_EXTS = {".png", ".jpg", ".jpeg"}
DEFAULT_NUM_AUG = 3
# ROTATE_RANGE = (-10, 10)
TRANSLATE_RANGE = (0, 0)
SCALE_RANGE = (1.0, 1.2)
FLIP_PROB = 0.4
NOISE_PROB = 0.5
BLUR_PROB = 0.25
SEED = 42

random.seed(SEED)
np.random.seed(SEED)


def parse_args():
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Augment YOLO training images and labels for esophagus dataset.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(os.environ.get("DATA_ROOT", repo_root / "dataset" / "esophagus_resized_yolo_nopadding")),
        help="Dataset root directory containing images/train and labels/train",
    )
    parser.add_argument(
        "--num-aug",
        type=int,
        default=DEFAULT_NUM_AUG,
        help="Number of augmentations to generate per source image",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Random seed for reproducible augmentation",
    )
    return parser.parse_args()


def augment_image(img: np.ndarray) -> np.ndarray:
    aug = img.astype(np.float32).copy()

    if random.random() < NOISE_PROB:
        noise = np.random.normal(0, 5, aug.shape).astype(np.float32)
        aug += noise

    aug = np.clip(aug, 0, 255).astype(np.uint8)

    if random.random() < BLUR_PROB:
        ksize = random.choice([3, 5])
        aug = cv2.GaussianBlur(aug, (ksize, ksize), 0)

    return aug


def get_affine_transform(img_w: int, img_h: int) -> np.ndarray:
    tx = random.uniform(*TRANSLATE_RANGE)
    ty = random.uniform(*TRANSLATE_RANGE)
    scale = random.uniform(*SCALE_RANGE)

    cx, cy = img_w / 2.0, img_h / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), 0, scale)
    M[0, 2] += tx
    M[1, 2] += ty
    return M


def transform_bbox(cx: float, cy: float, bw: float, bh: float, M: np.ndarray, img_w: int, img_h: int) -> tuple[float, float, float, float]:
    x1 = (cx - bw / 2.0) * img_w
    y1 = (cy - bh / 2.0) * img_h
    x2 = (cx + bw / 2.0) * img_w
    y2 = (cy + bh / 2.0) * img_h

    corners = np.array([
        [x1, y1, 1],
        [x2, y1, 1],
        [x2, y2, 1],
        [x1, y2, 1],
    ], dtype=np.float64)

    transformed = (M @ corners.T).T

    new_x1 = transformed[:, 0].min()
    new_y1 = transformed[:, 1].min()
    new_x2 = transformed[:, 0].max()
    new_y2 = transformed[:, 1].max()

    new_x1 = max(0, min(img_w, new_x1))
    new_y1 = max(0, min(img_h, new_y1))
    new_x2 = max(0, min(img_w, new_x2))
    new_y2 = max(0, min(img_h, new_y2))

    new_cx = ((new_x1 + new_x2) / 2.0) / img_w
    new_cy = ((new_y1 + new_y2) / 2.0) / img_h
    new_bw = (new_x2 - new_x1) / img_w
    new_bh = (new_y2 - new_y1) / img_h

    return new_cx, new_cy, new_bw, new_bh


def get_image_paths(img_dir: Path) -> list[Path]:
    return sorted([p for ext in IMG_EXTS for p in img_dir.glob(f"*{ext}")])


def load_labels(lbl_path: Path) -> list[tuple[int, float, float, float, float]]:
    labels = []
    for line in lbl_path.read_text().strip().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls_id = int(parts[0])
        cx, cy, bw, bh = map(float, parts[1:5])
        labels.append((cls_id, cx, cy, bw, bh))
    return labels


def augment_dataset(data_root: Path, num_aug: int, seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)

    img_dir = data_root / "images" / "train"
    lbl_dir = data_root / "labels" / "train"

    if not img_dir.exists() or not lbl_dir.exists():
        raise FileNotFoundError(f"Missing dataset directories: {img_dir} or {lbl_dir}")

    img_files = get_image_paths(img_dir)
    print(f"[INFO] Dataset root: {data_root}")
    print(f"[INFO] Found {len(img_files)} training images, generating {num_aug} augmentations each")

    total_generated = 0
    for img_path in img_files:
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        labels = load_labels(lbl_path) if lbl_path.exists() else []
        is_background = len(labels) == 0

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        img_h, img_w = img.shape[:2]

        for i in range(num_aug):
            M = get_affine_transform(img_w, img_h)
            aug_img = cv2.warpAffine(
                img,
                M,
                (img_w, img_h),
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(114, 114, 114),
            )
            aug_img = augment_image(aug_img)

            flipped = False
            if random.random() < FLIP_PROB:
                aug_img = cv2.flip(aug_img, 1)
                flipped = True

            new_labels = []
            for cls_id, cx, cy, bw, bh in labels:
                ncx, ncy, nbw, nbh = transform_bbox(cx, cy, bw, bh, M, img_w, img_h)
                if flipped:
                    ncx = 1.0 - ncx

                if nbw < 0.01 or nbh < 0.01:
                    continue
                if ncx < 0.0 or ncx > 1.0 or ncy < 0.0 or ncy > 1.0:
                    continue

                new_labels.append(f"{cls_id} {ncx:.6f} {ncy:.6f} {nbw:.6f} {nbh:.6f}")

            if not new_labels and not is_background:
                continue

            aug_name = f"{img_path.stem}_aug{i}"
            cv2.imwrite(str(img_dir / f"{aug_name}.png"), aug_img)
            (lbl_dir / f"{aug_name}.txt").write_text("\n".join(new_labels) + ("\n" if new_labels else ""))
            total_generated += 1

    print(f"[DONE] Generated {total_generated} augmented samples, total training images now: {len(get_image_paths(img_dir))}")


def main():
    args = parse_args()
    augment_dataset(args.data_root, args.num_aug, args.seed)


if __name__ == "__main__":
    main()
