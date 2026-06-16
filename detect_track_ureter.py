"""
YOLOv11 Detection + OSTrack Tracking Pipeline for Pylorus
- Frame 0: YOLO detects targets, initializes OSTrack trackers
- Frame 1~N: OSTrack tracks, YOLO re-detects every K frames or when score drops
"""
import sys
import os
import cv2
import numpy as np
import torch
from pathlib import Path
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parent
OSTRACK_ROOT = os.environ.get("OSTRACK_ROOT", str(REPO_ROOT / "external" / "OSTrack"))
sys.path.insert(0, OSTRACK_ROOT)

from lib.models.ostrack import build_ostrack
from lib.test.tracker.data_utils import Preprocessor
from lib.train.data.processing_utils import sample_target
from lib.utils.box_ops import clip_box
from lib.utils.ce_utils import generate_mask_cond
from lib.test.utils.hann import hann2d
from lib.config.ostrack.config import cfg, update_config_from_file

# ===================== CONFIG =====================
YOLO_WEIGHTS = os.environ.get("YOLO_WEIGHTS", str(REPO_ROOT / "weights" / "ureter_best.pt"))
OSTRACK_CKPT = os.environ.get("OSTRACK_CKPT", str(Path(OSTRACK_ROOT) / "checkpoints" / "pytorch_model.bin"))
OSTRACK_YAML = os.path.join(OSTRACK_ROOT, "experiments/ostrack/vitb_384_mae_ce_32x4_ep300.yaml")

VIDEO_DIR = os.environ.get("VIDEO_DIR", str(REPO_ROOT / "data" / "videos" / "ureter"))
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", str(REPO_ROOT / "outputs" / "detect_track" / "ureter"))

YOLO_CONF = 0.6          # YOLO confidence threshold. Lower (0.1~0.2) = more detections but more false positives. Higher (0.5~0.7) = fewer but more reliable detections.
YOLO_IMGSZ = 480         # YOLO input image size. Should match training resolution. Larger = more accurate but slower.
REDETECT_INTERVAL = 5    # Run YOLO every N frames even if tracking is good. Lower = more corrections but slower. Higher = faster but may drift.
TRACK_SCORE_THRESH = 0.4  # OSTrack score below this triggers immediate YOLO re-detection. Lower (0.1) = tolerate poor tracking longer. Higher (0.4) = re-detect more aggressively.
REINIT_IOU_THRESH = 0.2  # If IoU between tracker box and YOLO box < this, assume tracker drifted and trust YOLO. Lower = stricter drift detection. Higher = more tolerant of disagreement.
FUSION_IOU_THRESH = 0.5  # If IoU >= this, YOLO and tracker agree -> weighted fusion. If IoU < this but >= REINIT_IOU_THRESH, still fuse but with lower tracker weight.
MIN_BOX_AREA = 20        # Minimum bbox area (w*h in pixels) to consider valid. Filters out tiny spurious detections.
DEVICE = [0, 1, 2, 3, 7]  # Multiple GPUs for parallel inference. Each GPU processes different videos concurrently.
MAX_VIDEOS = "all"       # Number of videos to process. Set to "all" for full run, or int for debugging.

# Kalman filter parameters
KF_PROCESS_NOISE = 1.0       # Q: process noise. Larger = Kalman trusts new measurements more (responsive). Smaller = smoother but may lag.
KF_MEASUREMENT_NOISE = 5.0   # R: measurement noise. Larger = Kalman smooths more aggressively (less jitter). Smaller = follows raw detections closely.

CLASS_NAMES = ['ureteral orifice', 'ureteral lumen', 'polyps']
# priority: polyps(2) > ureteral lumen(1) > ureteral orifice(0). When multiple classes detected, keep the highest priority one.
CLASS_PRIORITY = {2: 0, 1: 1, 0: 2}  # lower value = higher priority
CLASS_COLORS = [(0, 255, 0), (255, 0, 0), (0, 0, 255)]  # BGR colors for each class
# ================================================


class OSTrackWrapper:
    """Lightweight wrapper around OSTrack for single-object tracking."""

    def __init__(self, cfg, checkpoint_path, device_id=0):
        self.cfg = cfg
        self.device = torch.device(f'cuda:{device_id}')
        self.network = build_ostrack(cfg, training=False)

        ckpt = torch.load(checkpoint_path, map_location='cpu')
        if 'net' in ckpt:
            self.network.load_state_dict(ckpt['net'], strict=True)
        elif 'model' in ckpt:
            self.network.load_state_dict(ckpt['model'], strict=True)
        else:
            self.network.load_state_dict(ckpt, strict=True)

        self.network = self.network.to(self.device)
        self.network.eval()
        self.preprocessor = Preprocessor()

        self.feat_sz = cfg.TEST.SEARCH_SIZE // cfg.MODEL.BACKBONE.STRIDE
        self.output_window = hann2d(torch.tensor([self.feat_sz, self.feat_sz]).long(), centered=True).to(self.device)

        self.template_factor = cfg.TEST.TEMPLATE_FACTOR
        self.template_size = cfg.TEST.TEMPLATE_SIZE
        self.search_factor = cfg.TEST.SEARCH_FACTOR
        self.search_size = cfg.TEST.SEARCH_SIZE

    def initialize(self, image, bbox):
        self.state = list(bbox)
        z_patch_arr, resize_factor, z_amask_arr = sample_target(
            image, self.state, self.template_factor, output_sz=self.template_size
        )
        template = self.preprocessor.process(z_patch_arr, z_amask_arr)
        with torch.no_grad():
            self.z_dict1 = template

        self.box_mask_z = None
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            template_bbox = self._transform_bbox_to_crop(
                self.state, resize_factor, template.tensors.device
            ).squeeze(1)
            self.box_mask_z = generate_mask_cond(self.cfg, 1, template.tensors.device, template_bbox)

    def track(self, image):
        H, W, _ = image.shape
        x_patch_arr, resize_factor, x_amask_arr = sample_target(
            image, self.state, self.search_factor, output_sz=self.search_size
        )
        search = self.preprocessor.process(x_patch_arr, x_amask_arr)

        with torch.no_grad():
            out_dict = self.network.forward(
                template=self.z_dict1.tensors,
                search=search.tensors,
                ce_template_mask=self.box_mask_z
            )

        pred_score_map = out_dict['score_map']
        response = self.output_window * pred_score_map
        score = response.max().item()

        pred_boxes = self.network.box_head.cal_bbox(response, out_dict['size_map'], out_dict['offset_map'])
        pred_boxes = pred_boxes.view(-1, 4)
        pred_box = (pred_boxes.mean(dim=0) * self.search_size / resize_factor).tolist()

        cx_prev = self.state[0] + 0.5 * self.state[2]
        cy_prev = self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        new_box = [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

        self.state = clip_box(new_box, H, W, margin=10)
        return self.state, score

    def _transform_bbox_to_crop(self, bbox, resize_factor, device):
        x, y, w, h = bbox
        crop_sz = self.template_size
        cx_crop = crop_sz / 2.0
        cy_crop = crop_sz / 2.0
        x_crop = cx_crop - w * resize_factor / 2.0
        y_crop = cy_crop - h * resize_factor / 2.0
        w_crop = w * resize_factor
        h_crop = h * resize_factor
        return torch.tensor([x_crop, y_crop, w_crop, h_crop], device=device).unsqueeze(0).unsqueeze(0)


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


# PLACEHOLDER_REMAINING


class TrackedObject:
    """One tracked object with its class, tracker instance, and Kalman filter."""

    def __init__(self, cls_id, bbox, tracker_template):
        self.cls_id = cls_id
        self.bbox = bbox
        self.kf = BBoxKalmanFilter(bbox)
        self.smooth_bbox = list(bbox)
        self.tracker = tracker_template
        self.lost_count = 0

    def kalman_predict(self):
        self.smooth_bbox = self.kf.predict()
        return self.smooth_bbox

    def kalman_update(self, measurement):
        self.kf.predict()
        self.smooth_bbox = self.kf.update(measurement)
        self.bbox = measurement
        return self.smooth_bbox


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


def fuse_boxes(yolo_box, track_box, yolo_conf, track_score):
    """Weighted fusion of YOLO and tracker boxes based on their scores.
    Returns fused [x, y, w, h].
    """
    # normalize scores to get weights
    w_yolo = yolo_conf
    w_track = track_score
    total = w_yolo + w_track
    if total < 1e-6:
        return yolo_box
    alpha = w_yolo / total  # YOLO weight
    fused = [
        alpha * yolo_box[i] + (1 - alpha) * track_box[i]
        for i in range(4)
    ]
    return fused


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


def process_video(video_path, yolo_model, ostrack_cfg, ostrack_ckpt, output_dir, device_id=0):
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

    # Resume: check if txt file has entries covering all frames
    if txt_path.exists():
        last_frame = -1
        with open(txt_path, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if parts:
                    last_frame = max(last_frame, int(parts[0]))
        if last_frame >= total_frames - 1:
            print(f"  [SKIP] {video_path.name} (already completed, {last_frame + 1} frames)")
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

        if tracked_obj is None:
            det = yolo_detect_best(yolo_model, frame, device_id)
            if det is not None:
                cls_id, bbox, conf = det
                tracker = OSTrackWrapper(ostrack_cfg, ostrack_ckpt, device_id)
                tracker.initialize(frame, bbox)
                tracked_obj = TrackedObject(cls_id, bbox, tracker)
        else:
            predicted_box = tracked_obj.kalman_predict()

            track_box = None
            track_score = 0.0
            try:
                track_box, track_score = tracked_obj.tracker.track(frame)
            except Exception:
                track_box = None

            if track_box is not None:
                bw, bh = track_box[2], track_box[3]
                if bw * bh < MIN_BOX_AREA:
                    track_box = None

            need_yolo = (
                (frame_id % REDETECT_INTERVAL == 0)
                or (track_box is None)
                or (track_score < TRACK_SCORE_THRESH)
            )

            yolo_box = None
            yolo_cls = None
            yolo_conf = 0.0
            if need_yolo:
                det = yolo_detect_best(yolo_model, frame, device_id)
                if det is not None:
                    yolo_cls, yolo_box, yolo_conf = det

            # Fusion decision
            if yolo_box is not None:
                use_yolo = True
                if yolo_cls != tracked_obj.cls_id:
                    if CLASS_PRIORITY.get(yolo_cls, 99) > CLASS_PRIORITY.get(tracked_obj.cls_id, 99):
                        use_yolo = False

                if use_yolo:
                    if track_box is None:
                        # tracker failed, use YOLO only
                        final_measure = yolo_box
                        tracked_obj.tracker.initialize(frame, final_measure)
                    else:
                        overlap = iou(track_box, yolo_box)
                        if overlap < REINIT_IOU_THRESH:
                            # severe disagreement: tracker drifted, trust YOLO and reinit
                            final_measure = yolo_box
                            tracked_obj.tracker.initialize(frame, final_measure)
                        elif overlap >= FUSION_IOU_THRESH:
                            # high agreement: weighted fusion of both
                            final_measure = fuse_boxes(yolo_box, track_box, yolo_conf, track_score)
                            tracked_obj.tracker.initialize(frame, final_measure)
                        else:
                            # moderate disagreement: fuse but give YOLO more weight
                            boosted_yolo_conf = min(1.0, yolo_conf * 1.5)
                            final_measure = fuse_boxes(yolo_box, track_box, boosted_yolo_conf, track_score)
                            tracked_obj.tracker.initialize(frame, final_measure)

                    tracked_obj.cls_id = yolo_cls
                    tracked_obj.lost_count = 0
                    tracked_obj.kalman_update(final_measure)
                else:
                    # YOLO found lower priority class, keep tracking current target
                    if track_box is not None:
                        tracked_obj.kalman_update(track_box)
                    else:
                        tracked_obj.smooth_bbox = predicted_box
                        tracked_obj.lost_count += 1
            elif track_box is not None:
                # no YOLO result, use tracker alone
                tracked_obj.kalman_update(track_box)
                tracked_obj.lost_count = 0
            else:
                # both failed, use Kalman prediction only
                tracked_obj.smooth_bbox = predicted_box
                tracked_obj.lost_count += 1

            if tracked_obj.lost_count >= 5:
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


def worker(gpu_id, video_list, ostrack_ckpt, output_dir):
    """Worker function: one GPU processes a subset of videos."""
    torch.cuda.set_device(gpu_id)
    print(f"[GPU {gpu_id}] Loading models, assigned {len(video_list)} videos")

    yolo_model = YOLO(YOLO_WEIGHTS)
    update_config_from_file(OSTRACK_YAML)

    for video_path in video_list:
        process_video(video_path, yolo_model, cfg, ostrack_ckpt, output_dir, device_id=gpu_id)

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

    # distribute videos across GPUs round-robin
    gpu_video_map = {gpu_id: [] for gpu_id in DEVICE}
    for i, video_path in enumerate(videos):
        gpu_id = DEVICE[i % len(DEVICE)]
        gpu_video_map[gpu_id].append(video_path)

    for gpu_id, vids in gpu_video_map.items():
        print(f"  GPU {gpu_id}: {len(vids)} videos")

    # launch one process per GPU
    processes = []
    for gpu_id, video_list in gpu_video_map.items():
        if not video_list:
            continue
        p = mp.Process(target=worker, args=(gpu_id, video_list, OSTRACK_CKPT, output_dir))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print(f"\n[DONE] All results saved to: {output_dir}")


if __name__ == "__main__":
    main()
