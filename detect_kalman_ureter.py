"""
YOLOv11 Detection + Kalman Filter Tracking Pipeline for Ureter
- Every frame: YOLO detects targets
- Kalman filter smooths bounding boxes across frames
- No OSTrack dependency
"""
import sys
import os
import cv2
import numpy as np
import torch
from pathlib import Path
from ultralytics import YOLO

# ===================== CONFIG =====================
REPO_ROOT = Path(__file__).resolve().parent
YOLO_WEIGHTS = os.environ.get("YOLO_WEIGHTS", str(REPO_ROOT / "weights" / "ureter_best.pt"))

VIDEO_DIR = os.environ.get("VIDEO_DIR", str(REPO_ROOT / "data" / "videos" / "ureter"))
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", str(REPO_ROOT / "outputs" / "detect_kalman" / "ureter"))

YOLO_CONF = 0.4
YOLO_IMGSZ = 480
DETECT_EVERY_FRAME = True   # True = run YOLO every frame; False = run every REDETECT_INTERVAL frames
REDETECT_INTERVAL = 1       # Only used if DETECT_EVERY_FRAME=False
MIN_BOX_AREA = 20
MAX_LOST_FRAMES = 10        # After this many frames without detection, drop the target
DEVICE = [0, 1, 2]
MAX_VIDEOS = "all"
# MAX_VIDEOS = 10

# Kalman filter parameters
KF_PROCESS_NOISE = 1.0
KF_MEASUREMENT_NOISE = 30.0

# EMA smoothing for center position (post-processing after Kalman)
EMA_ALPHA = 0.2  # 0~1, smaller = smoother but more lag, larger = responsive but less smooth

CLASS_NAMES = ['ureteral orifice', 'ureteral lumen', 'polyps']
CLASS_PRIORITY = {2: 0, 1: 1, 0: 2}  # lower value = higher priority
CLASS_COLORS = [(0, 255, 0), (255, 0, 0), (0, 0, 255)]  # BGR
# ================================================


class BBoxKalmanFilter:
    """Kalman filter for bbox [x, y, w, h] with constant velocity model."""

    def __init__(self, bbox):
        self.kf = cv2.KalmanFilter(8, 4)
        self.kf.transitionMatrix = np.eye(8, dtype=np.float32)
        for i in range(4):
            self.kf.transitionMatrix[i, i + 4] = 1.0
        self.kf.measurementMatrix = np.zeros((4, 8), dtype=np.float32)
        for i in range(4):
            self.kf.measurementMatrix[i, i] = 1.0
        self.kf.processNoiseCov = np.eye(8, dtype=np.float32) * KF_PROCESS_NOISE
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * KF_MEASUREMENT_NOISE
        self.kf.errorCovPost = np.eye(8, dtype=np.float32)
        self.kf.statePost = np.zeros((8, 1), dtype=np.float32)
        for i in range(4):
            self.kf.statePost[i, 0] = bbox[i]

    def predict(self):
        state = self.kf.predict()
        return [float(state[i, 0]) for i in range(4)]

    def update(self, bbox):
        measurement = np.array([[bbox[0]], [bbox[1]], [bbox[2]], [bbox[3]]], dtype=np.float32)
        self.kf.correct(measurement)
        return [float(self.kf.statePost[i, 0]) for i in range(4)]


class TrackedObject:
    """One tracked object with its class and Kalman filter."""

    def __init__(self, cls_id, bbox):
        self.cls_id = cls_id
        self.bbox = bbox
        self.kf = BBoxKalmanFilter(bbox)
        self.smooth_bbox = list(bbox)
        self.lost_count = 0
        # EMA state for center position
        self.ema_cx = bbox[0] + bbox[2] / 2.0
        self.ema_cy = bbox[1] + bbox[3] / 2.0

    def kalman_predict(self):
        self.smooth_bbox = self.kf.predict()
        self._apply_ema()
        return self.smooth_bbox

    def kalman_update(self, measurement):
        self.kf.predict()
        self.smooth_bbox = self.kf.update(measurement)
        self._apply_ema()
        self.bbox = measurement
        return self.smooth_bbox

    def _apply_ema(self):
        """Apply EMA on center position, keep w/h from Kalman directly."""
        x, y, w, h = self.smooth_bbox
        cx = x + w / 2.0
        cy = y + h / 2.0
        self.ema_cx = EMA_ALPHA * cx + (1 - EMA_ALPHA) * self.ema_cx
        self.ema_cy = EMA_ALPHA * cy + (1 - EMA_ALPHA) * self.ema_cy
        self.smooth_bbox = [self.ema_cx - w / 2.0, self.ema_cy - h / 2.0, w, h]


def yolo_detect_best(model, frame, device_id=0):
    """Run YOLO detection, return the single best detection by priority."""
    results = model.predict(frame, conf=YOLO_CONF, imgsz=YOLO_IMGSZ, device=device_id,
                            half=True, verbose=False)
    detections = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            w, h = x2 - x1, y2 - y1
            if w * h < MIN_BOX_AREA:
                continue
            detections.append((cls_id, [float(x1), float(y1), float(w), float(h)], conf))
    if not detections:
        return None
    detections.sort(key=lambda d: (CLASS_PRIORITY.get(d[0], 99), -d[2]))
    return detections[0]


def iou(box1, box2):
    """Compute IoU between two [x,y,w,h] boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[0] + box1[2], box2[0] + box2[2])
    y2 = min(box1[1] + box1[3], box2[1] + box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = box1[2] * box1[3]
    area2 = box2[2] * box2[3]
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0


def draw_results(frame, tracked_obj, frame_id):
    vis = frame.copy()
    if tracked_obj is not None:
        x, y, w, h = [int(v) for v in tracked_obj.smooth_bbox]
        color = CLASS_COLORS[tracked_obj.cls_id % len(CLASS_COLORS)]
        cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
        label = f"{CLASS_NAMES[tracked_obj.cls_id]}"
        cv2.putText(vis, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    cv2.putText(vis, f"Frame: {frame_id}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return vis


def process_video(video_path, yolo_model, output_dir, device_id=0):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[WARN] Cannot open: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_video_dir = output_dir / "videos"
    out_txt_dir = output_dir / "tracks"
    out_video_dir.mkdir(parents=True, exist_ok=True)
    out_txt_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_video_dir / (video_path.stem + "_tracked.mp4")
    txt_path = out_txt_dir / (video_path.stem + "_tracks.txt")

    if txt_path.exists():
        last_frame = -1
        with open(txt_path, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if parts:
                    last_frame = max(last_frame, int(parts[0]))
        if last_frame >= total_frames - 1:
            print(f"  [SKIP] {video_path.name} (already completed)")
            cap.release()
            return

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
    txt_file = open(txt_path, 'w')

    tracked_obj = None
    frame_id = 0

    print(f"  Processing: {video_path.name} ({total_frames} frames, {w}x{h}, {fps:.1f}fps)")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        need_detect = DETECT_EVERY_FRAME or (frame_id % REDETECT_INTERVAL == 0) or (tracked_obj is None)

        if need_detect:
            det = yolo_detect_best(yolo_model, frame, device_id)
        else:
            det = None

        if tracked_obj is None:
            if det is not None:
                cls_id, bbox, conf = det
                tracked_obj = TrackedObject(cls_id, bbox)
        else:
            predicted_box = tracked_obj.kalman_predict()

            if det is not None:
                cls_id, bbox, conf = det
                use_det = True
                if cls_id != tracked_obj.cls_id:
                    if CLASS_PRIORITY.get(cls_id, 99) > CLASS_PRIORITY.get(tracked_obj.cls_id, 99):
                        use_det = False

                if use_det:
                    overlap = iou(predicted_box, bbox)
                    if overlap > 0.1:
                        tracked_obj.kalman_update(bbox)
                    else:
                        tracked_obj = TrackedObject(cls_id, bbox)
                    tracked_obj.cls_id = cls_id
                    tracked_obj.lost_count = 0
                else:
                    tracked_obj.smooth_bbox = predicted_box
                    tracked_obj.lost_count += 1
            else:
                tracked_obj.smooth_bbox = predicted_box
                tracked_obj.lost_count += 1

            if tracked_obj.lost_count >= MAX_LOST_FRAMES:
                tracked_obj = None

        if tracked_obj is not None:
            x, y, bw, bh = tracked_obj.smooth_bbox
            txt_file.write(f"{frame_id},{tracked_obj.cls_id},{x:.1f},{y:.1f},{bw:.1f},{bh:.1f}\n")

        vis = draw_results(frame, tracked_obj, frame_id)
        writer.write(vis)
        frame_id += 1

        if frame_id % 100 == 0:
            obj_info = f"tracking {CLASS_NAMES[tracked_obj.cls_id]}" if tracked_obj else "no target"
            print(f"    Frame {frame_id}/{total_frames}, {obj_info}")

    cap.release()
    writer.release()
    txt_file.close()
    print(f"  Saved: {out_path}")


def worker(gpu_id, video_list, output_dir):
    """Worker function: one GPU processes a subset of videos."""
    torch.cuda.set_device(gpu_id)
    print(f"[GPU {gpu_id}] Loading YOLO model, assigned {len(video_list)} videos")

    yolo_model = YOLO(YOLO_WEIGHTS)

    for video_path in video_list:
        process_video(video_path, yolo_model, output_dir, device_id=gpu_id)

    print(f"[GPU {gpu_id}] Done.")


def main():
    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_dir = Path(VIDEO_DIR)
    videos = sorted(video_dir.glob("*.mp4"))

    if MAX_VIDEOS != "all":
        videos = videos[:int(MAX_VIDEOS)]

    print(f"[INFO] Found {len(videos)} videos, distributing across {len(DEVICE)} GPUs: {DEVICE}")

    gpu_video_map = {gpu_id: [] for gpu_id in DEVICE}
    for i, video_path in enumerate(videos):
        gpu_id = DEVICE[i % len(DEVICE)]
        gpu_video_map[gpu_id].append(video_path)

    for gpu_id, vids in gpu_video_map.items():
        print(f"  GPU {gpu_id}: {len(vids)} videos")

    processes = []
    for gpu_id, video_list in gpu_video_map.items():
        if not video_list:
            continue
        p = mp.Process(target=worker, args=(gpu_id, video_list, output_dir))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print(f"\n[DONE] All results saved to: {output_dir}")


if __name__ == "__main__":
    main()
