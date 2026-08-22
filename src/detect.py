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
# COCO 80类中选取监控场景常见对象：人 + 常见动物（猫/狗/马/鸟/羊/牛）
OBJ_CLASSES = {
    0: "person",
    14: "bird",    # 鸟/鸡/鸭等家禽
    15: "cat",
    16: "dog",
    17: "horse",   # 马
    18: "sheep",   # 羊
    19: "cow",     # 牛
}


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
    """Heuristic pose label from COCO-17 keypoints: lie/stand/sit/squat/bend/unknown.

    基于 COCO-17 关键点几何的单帧姿态分类（无额外模型）：
    - lie: 身体水平（合并原 fall 跌倒）
    - squat: 髋部接近脚踝（下蹲）
    - sit: 大腿水平（坐）
    - bend: 躯干缩短但站立（弯腰）
    - stand: 默认站立
    - unknown: 关键点不足无法判断
    """
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

    # --- lie 判断（严格，避免站姿误判）---
    # 必要条件 1：bounding box 横向明显长于纵向（ bw > bh * 1.4 ）
    # 必要条件 2：肩膀中点与髋部中点 y 坐标接近（水平姿态）
    # 必要条件 3：脚踝 y 坐标接近肩膀/髋部 y（不是远低于髋部）
    if bw > bh * 1.4 and 5 in pts and 6 in pts and 11 in pts and 12 in pts:
        sho_y = (pts[5][1] + pts[6][1]) / 2
        hip_y = (pts[11][1] + pts[12][1]) / 2
        # 肩-髋垂直距离应远小于身体高度（水平躺平时肩髋接近水平）
        if abs(sho_y - hip_y) < bh * 0.35:
            # 进一步检查脚踝：躺平时脚踝不应远低于髋部
            ank_y_min = min(
                pts[15][1] if 15 in pts else hip_y,
                pts[16][1] if 16 in pts else hip_y
            )
            if ank_y_min - hip_y < bh * 0.5:
                return "lie"

    if not (5 in pts and 6 in pts and 11 in pts and 12 in pts
            and 15 in pts and 16 in pts):
        return "unknown"
    sho_x = (pts[5][0] + pts[6][0]) / 2
    sho_y = (pts[5][1] + pts[6][1]) / 2
    hip_x = (pts[11][0] + pts[12][0]) / 2
    hip_y = (pts[11][1] + pts[12][1]) / 2
    ank_y = (pts[15][1] + pts[16][1]) / 2
    knee_y = ((pts[13][1] if 13 in pts else hip_y)
              + (pts[14][1] if 14 in pts else ank_y)) / 2
    # 蹲：髋部接近脚踝（垂直压缩，阈值放宽到 0.25*bh 容纳浅蹲）
    if ank_y - hip_y < 0.25 * bh:
        return "squat" if knee_y > hip_y else "unknown"
    thigh = knee_y - hip_y
    shin = ank_y - knee_y
    if thigh <= 0:
        # 膝盖与髋部同高或更高 → 大腿水平 → 坐姿
        return "sit"
    if shin <= 0:
        return "stand"
    # 坐：大腿水平（thigh 远短于 shin）
    if thigh / shin < 0.6:
        return "sit"
    # 弯腰：躯干在图像上垂直投影明显缩短（肩膀下移接近髋部），
    # 但腿仍直立。torso/leg 比值小表示躯干前倾。
    torso = hip_y - sho_y
    leg = ank_y - hip_y
    if leg > 0 and torso / leg < 0.45:
        return "bend"
    return "stand"


def _pose_from_result(pr):
    """从 pose 模型结果提取第一个 person 的 (keypoints, kconf, pose_label)。
    返回 (kp, kconf, label) 或 (None, None, "unknown")。"""
    if (pr.keypoints is None or not hasattr(pr.keypoints, 'xy')
            or pr.keypoints.xy is None or len(pr.keypoints.xy) == 0
            or len(pr.keypoints.xy[0]) == 0):
        return None, None, "unknown"
    kp = pr.keypoints.xy[0].cpu().numpy()
    kconf = (pr.keypoints.conf[0].cpu().numpy()
             if (pr.keypoints.conf is not None and len(pr.keypoints.conf) > 0)
             else None)
    return kp, kconf, classify_pose(kp, kconf)


def _crop_person(img_path, bbox, padding=0.3, min_size=320):
    """围绕 person bbox 裁剪并上采样，返回裁剪后的 numpy 图像（BGR）。

    padding: 相对 bbox 较大边的扩展比例（0.3 = 扩 30%），避免切掉肢体。
    min_size: 裁剪后最短边小于此值则上采样，保证 pose 模型输入分辨率。
    """
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    h, w = img.shape[:2]
    x1, y1, x2, y2 = bbox
    bw = x2 - x1
    bh = y2 - y1
    side = max(bw, bh)
    pad = side * padding
    cx1 = max(0, int(x1 - pad))
    cy1 = max(0, int(y1 - pad))
    cx2 = min(w, int(x2 + pad))
    cy2 = min(h, int(y2 + pad))
    crop = img[cy1:cy2, cx1:cx2]
    ch, cw = crop.shape[:2]
    shortest = min(ch, cw)
    if shortest < min_size and shortest > 0:
        scale = min_size / shortest
        crop = cv2.resize(crop, (int(cw * scale), int(ch * scale)),
                          interpolation=cv2.INTER_CUBIC)
    return crop


def detect_objects(paths, person_conf=None, pet_conf=None, pose=None, batch=16,
                   progress_cb=None, video_name=""):
    """Detect person/animals across image paths. Returns a list (per input) of
    dicts: {obj_type, score, bbox, keypoints, kconf, pose} or None.

    支持的对象（COCO 类）：person / bird(鸡鸭) / cat / dog / horse / sheep / cow
    优化策略（两者结合）：
    1. 人物直接用 pose_model 一体推理（检测+关键点），省去 det_model 二次推理
    2. 动物用 det_model 多类检测
    3. 人物 pose=unknown 且 bbox 较小时，裁剪+上采样后重新跑 pose_model

    progress_cb: 可选回调，每完成一批检测/一次 pose 推理时调用，用于前端
    展示细粒度进度（当前第几帧、识别到什么对象等）。
    video_name: 当前视频名，仅用于进度描述。
    """
    person_conf = person_conf if person_conf is not None else config.DET_CONF
    pet_conf = pet_conf if pet_conf is not None else config.PET_CONF
    if pose is None:
        pose = config.USE_POSE

    total_frames = len(paths)
    t0 = time.time()
    print(f"[detect] starting: {total_frames} frames, "
          f"person_conf={person_conf}, pet_conf={pet_conf}, pose={pose}, "
          f"batch={batch}")

    det_model = load_model(pose=False)  # 多类检测器（人/动物）
    pose_model = load_model(pose=True) if pose else None  # 人物姿态一体模型
    out = []
    detected_count = 0
    retry_count = 0
    processed = 0  # 已处理的帧数（含本批）
    for i in range(0, total_frames, batch):
        chunk = paths[i:i + batch]
        results = det_model(chunk, classes=list(OBJ_CLASSES.keys()),
                            conf=min(person_conf, pet_conf), verbose=False)
        for r, p in zip(results, chunk):
            best = _best_multi(r, person_conf, pet_conf)
            # 取本帧文件名（用于 pose 进度描述）
            try:
                frame_label = Path(p).name
            except Exception:
                frame_label = "?"
            if best is None:
                out.append(None)
                processed += 1
                if progress_cb:
                    progress_cb({
                        "stage": "detect",
                        "text": f"识别中: {video_name}（{processed}/{total_frames} 帧）",
                        "current": processed, "total": total_frames,
                        "extra": {"video": video_name, "frame": frame_label,
                                  "detected_so_far": detected_count,
                                  "result": "无对象"},
                    })
                continue
            detected_count += 1
            obj_type, score, bbox = best
            kp = None
            kconf = None
            pose_label = "none"
            if obj_type == "person" and pose and pose_model is not None:
                # 一体推理：pose 模型本身能检测 person + 估计关键点
                if progress_cb:
                    progress_cb({
                        "stage": "pose",
                        "text": f"姿态识别: {video_name} · {frame_label}",
                        "current": processed + 1, "total": total_frames,
                        "extra": {"video": video_name, "frame": frame_label,
                                  "obj": obj_type, "phase": "pose_inference"},
                    })
                try:
                    pres = pose_model(p, classes=[0],
                                      conf=person_conf, verbose=False)
                    kp, kconf, pose_label = _pose_from_result(pres[0])
                    # 小人物 + pose=unknown → 裁剪+上采样重试
                    bw = bbox[2] - bbox[0]
                    bh = bbox[3] - bbox[1]
                    if pose_label == "unknown" and min(bw, bh) < 200:
                        if progress_cb:
                            progress_cb({
                                "stage": "pose",
                                "text": f"姿态裁剪重试: {video_name} · {frame_label}",
                                "current": processed + 1, "total": total_frames,
                                "extra": {"video": video_name, "frame": frame_label,
                                          "obj": obj_type, "phase": "crop_retry"},
                            })
                        crop = _crop_person(p, bbox)
                        if crop is not None:
                            try:
                                rpres = pose_model(crop, classes=[0],
                                                   conf=person_conf,
                                                   verbose=False)
                                rkp, rkconf, rlabel = _pose_from_result(rpres[0])
                                if rlabel != "unknown":
                                    pose_label = rlabel
                                    kp = rkp
                                    kconf = rkconf
                                    retry_count += 1
                            except Exception as e:
                                print(f"[detect] crop retry failed for {p}: {e}")
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
            processed += 1
            if progress_cb:
                progress_cb({
                    "stage": "detect",
                    "text": f"识别中: {video_name}（{processed}/{total_frames} 帧）",
                    "current": processed, "total": total_frames,
                    "extra": {"video": video_name, "frame": frame_label,
                              "detected_so_far": detected_count,
                              "result": f"{obj_type}/{pose_label}"},
                })

    elapsed = time.time() - t0
    # 统计检测到的对象类型分布
    type_counts = {}
    for d in out:
        if d:
            t = d["obj_type"]
            type_counts[t] = type_counts.get(t, 0) + 1
    type_str = ", ".join(f"{k}={v}" for k, v in sorted(type_counts.items())) or "none"
    print(f"[detect] done in {elapsed:.1f}s: {detected_count}/{total_frames} frames "
          f"had detections ({type_str}), pose_crop_retry={retry_count}")
    return out
