"""Person detection with YOLOv8 (Ultralytics).

Replaces the old OpenCV MobileNet-SSD detector: YOLOv8 is more accurate for
distant / small people, and the same model can be swapped for the pose variant
(`yolov8n-pose.pt`) for keypoint / posture information. Person = class 0.
"""
import time
import cv2
import numpy as np
from pathlib import Path

from . import config

_model = None
_pose_model = None


def load_model(pose=False):
    global _model, _pose_model
    from ultralytics import YOLO
    if pose:
        if _pose_model is None:
            t0 = time.time()
            print(f"[detect] loading pose model: {config.POSE_MODEL}")
            _pose_model = YOLO(str(config.POSE_MODEL))
            print(f"[detect] pose model loaded in {time.time()-t0:.1f}s")
        return _pose_model
    if _model is None:
        t0 = time.time()
        print(f"[detect] loading detection model: {config.MODEL_PATH}")
        _model = YOLO(str(config.MODEL_PATH))
        print(f"[detect] detection model loaded in {time.time()-t0:.1f}s")
    return _model


def _best_box(result, conf):
    """Return (score, (x1,y1,x2,y2)) of the highest-confidence person box."""
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return None
    best = None
    for b in boxes:
        if int(b.cls[0]) != 0:  # class 0 = person
            continue
        score = float(b.conf[0])
        if score < conf:
            continue
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        if best is None or score > best[0]:
            best = (score, (int(x1), int(y1), int(x2), int(y2)))
    return best


def detect_frames(paths, conf=None, pose=False, batch=16):
    """Run detection on a list of image/video paths. Returns a list of
    (score, bbox) per input, or None when no person is found."""
    conf = conf if conf is not None else config.DET_CONF
    model = load_model(pose=pose)
    out = []
    for i in range(0, len(paths), batch):
        chunk = paths[i:i + batch]
        results = model(chunk, classes=[0], conf=conf, verbose=False)
        for r in results:
            out.append(_best_box(r, conf))
    return out


def has_person(frame_or_path, conf=None, pose=False):
    """Return True if a person is detected in a single frame/image path."""
    conf = conf if conf is not None else config.DET_CONF
    model = load_model(pose=pose)
    res = model(frame_or_path, classes=[0], conf=conf, verbose=False)
    return _best_box(res[0], conf) is not None


def detect_person_best(frame_or_path, conf=None, pose=False):
    """Return (score, bbox) for the most confident person, or None."""
    conf = conf if conf is not None else config.DET_CONF
    model = load_model(pose=pose)
    res = model(frame_or_path, classes=[0], conf=conf, verbose=False)
    return _best_box(res[0], conf)


# COCO 17-keypoint skeleton (used by YOLOv8-pose) for drawing limbs.
POSE_SKELETON = [
    (5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16), (0, 1), (1, 3), (0, 2), (2, 4),
]


def detect_pose(paths, conf=None, batch=16):
    """Run the pose model on image/video paths. Returns a list of
    (score, xy, kconf) per input, where `xy` is a (17,2) keypoint array and
    `kconf` a (17,) confidence array, or None when no person is found.
    Requires the pose model (USE_POSE / POSE_MODEL)."""
    conf = conf if conf is not None else config.DET_CONF
    model = load_model(pose=True)
    out = []
    for i in range(0, len(paths), batch):
        chunk = paths[i:i + batch]
        results = model(chunk, classes=[0], conf=conf, verbose=False)
        for r in results:
            kp = r.keypoints
            if kp is None or len(kp.xy) == 0:
                out.append(None)
                continue
            xy = kp.xy[0].cpu().numpy()
            kconf = kp.conf[0].cpu().numpy() if kp.conf is not None else None
            best = _best_box(r, conf)
            score = best[0] if best else 0.0
            out.append((score, xy, kconf))
    return out


def draw_skeleton(image, xy, kconf=None, color=(0, 255, 128), width=2):
    """Draw the COCO pose skeleton on a PIL image. Returns the image.
    `xy` is a (17,2) array in pixel coordinates of `image`."""
    from PIL import ImageDraw
    draw = ImageDraw.Draw(image)
    for i, j in POSE_SKELETON:
        if kconf is not None and (kconf[i] < 0.3 or kconf[j] < 0.3):
            continue
        xi, yi = xy[i]
        xj, yj = xy[j]
        if xi <= 0 or yi <= 0 or xj <= 0 or yj <= 0:
            continue
        draw.line([(int(xi), int(yi)), (int(xj), int(yj))],
                  fill=color, width=width)
    for i, (x, y) in enumerate(xy):
        if kconf is not None and kconf[i] < 0.3:
            continue
        if x <= 0 or y <= 0:
            continue
        draw.ellipse([int(x) - 3, int(y) - 3, int(x) + 3, int(y) + 3],
                    fill=color)
    return image


# YOLO class id -> object type used by the event dashboard.
OBJ_CLASSES = {0: "person", 15: "cat", 16: "dog"}


def _best_multi(result, person_conf, pet_conf):
    """Return (obj_type, score, (x1,y1,x2,y2)) for the highest-confidence
    person/cat/dog box, honoring per-class confidence thresholds."""
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return None
    best = None
    for b in boxes:
        cid = int(b.cls[0])
        if cid not in OBJ_CLASSES:
            continue
        conf = float(b.conf[0])
        thr = person_conf if cid == 0 else pet_conf
        if conf < thr:
            continue
        score = conf
        if best is None or score > best[1]:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            best = (OBJ_CLASSES[cid], score, (int(x1), int(y1), int(x2), int(y2)))
    return best


def classify_pose(kp, kconf):
    """Heuristic pose label from COCO-17 keypoints: fall/stand/sit/squat/unknown.

    This is a lightweight rule-based classifier (no extra model). It is
    intentionally tolerant: ambiguous poses fall back to 'unknown'."""
    ids = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    pts = {}
    for i in ids:
        if kconf is None or (i < len(kconf) and kconf[i] >= 0.3):
            pts[i] = kp[i]
    if len(pts) < 4:
        return "unknown"
    xs = [p[0] for p in pts.values()]
    ys = [p[1] for p in pts.values()]
    bw = max(xs) - min(xs)
    bh = max(ys) - min(ys)
    if bh <= 0:
        return "unknown"
    # Lying down: body bounding box is wider than tall.
    if bw > bh * 0.9:
        return "fall"
    if not (5 in pts and 6 in pts and 11 in pts and 12 in pts
            and 15 in pts and 16 in pts):
        return "unknown"
    sho_y = (pts[5][1] + pts[6][1]) / 2
    hip_y = (pts[11][1] + pts[12][1]) / 2
    ank_y = (pts[15][1] + pts[16][1]) / 2
    knee_y = ((pts[13][1] if 13 in pts else hip_y)
              + (pts[14][1] if 14 in pts else ank_y)) / 2
    # Compressed vertically (hips near ankles) -> squat/sit family.
    if ank_y - hip_y < 0.15 * bh:
        return "squat" if knee_y > hip_y else "unknown"
    thigh = knee_y - hip_y
    shin = ank_y - knee_y
    if thigh <= 0 or shin <= 0:
        return "stand"
    # Sitting: thighs roughly horizontal (thigh much shorter than shin).
    if thigh / shin < 0.6:
        return "sit"
    return "stand"


def detect_objects(paths, person_conf=None, pet_conf=None, pose=None, batch=16):
    """Detect person/cat/dog across image paths. Returns a list (per input) of
    dicts: {obj_type, score, bbox, keypoints, kconf, pose} or None.

    Persons get a pose label (fall/stand/sit/squat/unknown) when the pose model
    is enabled; cats/dogs always get pose 'none'."""
    person_conf = person_conf if person_conf is not None else config.DET_CONF
    pet_conf = pet_conf if pet_conf is not None else config.PET_CONF
    if pose is None:
        pose = config.USE_POSE

    t0 = time.time()
    print(f"[detect] starting: {len(paths)} frames, "
          f"person_conf={person_conf}, pet_conf={pet_conf}, pose={pose}, "
          f"batch={batch}")

    det_model = load_model(pose=False)  # multi-class detector (person/cat/dog)
    out = []
    detected_count = 0
    for i in range(0, len(paths), batch):
        chunk = paths[i:i + batch]
        results = det_model(chunk, classes=[0, 15, 16],
                            conf=min(person_conf, pet_conf), verbose=False)
        for r, p in zip(results, chunk):
            best = _best_multi(r, person_conf, pet_conf)
            if best is None:
                out.append(None)
                continue
            detected_count += 1
            obj_type, score, bbox = best
            kp = None
            kconf = None
            if obj_type == "person" and pose:
                try:
                    pres = load_model(pose=True)(p, classes=[0],
                                                conf=person_conf, verbose=False)
                    pr = pres[0]
                    if (pr.keypoints is not None and
                            hasattr(pr.keypoints, 'xy') and
                            pr.keypoints.xy is not None and
                            len(pr.keypoints.xy) > 0 and
                            len(pr.keypoints.xy[0]) > 0):
                        kp = pr.keypoints.xy[0].cpu().numpy()
                        kconf = (pr.keypoints.conf[0].cpu().numpy()
                                 if (pr.keypoints.conf is not None and
                                     len(pr.keypoints.conf) > 0)
                                 else None)
                        pose_label = classify_pose(kp, kconf)
                    else:
                        pose_label = "unknown"
                except Exception as e:
                    print(f"[detect] pose inference failed for {p}: {e}")
                    pose_label = "unknown"
            elif obj_type == "person":
                pose_label = "unknown"
            else:
                pose_label = "none"
            out.append({
                "obj_type": obj_type, "score": score, "bbox": bbox,
                "keypoints": kp, "kconf": kconf, "pose": pose_label,
            })

    elapsed = time.time() - t0
    # 统计检测到的对象类型分布
    type_counts = {}
    for d in out:
        if d:
            t = d["obj_type"]
            type_counts[t] = type_counts.get(t, 0) + 1
    type_str = ", ".join(f"{k}={v}" for k, v in sorted(type_counts.items())) or "none"
    print(f"[detect] done in {elapsed:.1f}s: {detected_count}/{len(paths)} frames "
          f"had detections ({type_str})")
    return out
