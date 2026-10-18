"""Compare FP16 vs INT8 quantized pose model accuracy & latency.

用业务真实画面（models/calib/）对比：
  1. 检测一致性：person 框数量、置信度差、框 IoU、关键点平均距离
  2. 推理性能：预热后多次推理的平均/最大时延（ms）

用法：
  python tools/verify_int8.py [--imgs models/calib] [--n 60] [--iters 30]
"""
import argparse
import time
from pathlib import Path

import numpy as np
from ultralytics import YOLO

REPO = Path(__file__).resolve().parent.parent
MODELS = REPO / "models"
FP16_DIR = MODELS / "yolov8n-pose_openvino_model"
INT8_DIR = MODELS / "yolov8n-pose_int8_openvino_model"


def _matched(a, b, iou_thr=0.5):
    """把两组检测框按 IoU 贪心配对，返回 (配对索引, 未配对框数)。"""
    import itertools
    pairs = []
    used_b = set()
    for i, ba in enumerate(a):
        best_j, best_iou = -1, 0.0
        for j, bb in enumerate(b):
            if j in used_b:
                continue
            iou = _iou(ba, bb)
            if iou > best_iou:
                best_j, best_iou = j, iou
        if best_j >= 0 and best_iou >= iou_thr:
            pairs.append((i, best_j, best_iou))
            used_b.add(best_j)
    return pairs, len(a) - len(pairs), len(b) - len(pairs)


def _iou(a, b):
    x1, y1 = np.maximum(a[:2], b[:2])
    x2, y2 = np.minimum(a[2:], b[2:])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / max(area_a + area_b - inter, 1e-6)


def _keypoint_dist(kp_fp16, kp_int8):
    """同名关键点之间的平均欧氏距离。kp 支持 (N,17,3) 或 (17,3)。"""
    if kp_fp16 is None or kp_int8 is None:
        return float("nan")
    a = np.asarray(kp_fp16, dtype=float).reshape(-1, 3)
    b = np.asarray(kp_int8, dtype=float).reshape(-1, 3)
    if a.shape[0] != b.shape[0]:
        return float("nan")
    vis = a[:, 2] > 0.5
    if vis.sum() == 0:
        return float("nan")
    d = np.linalg.norm(a[vis][:, :2] - b[vis][:, :2], axis=1)
    return float(d.mean())


def run(model, imgs, iters=20, conf=0.3):
    """在 imgs 上推理（预热后计时），返回 (每张的 boxes/keypoints/conf, 时延列表)。"""
    lat = []
    # 预热
    model(imgs[0], conf=conf, verbose=False)
    for i in range(iters):
        t0 = time.perf_counter()
        r = model(imgs[i % len(imgs)], conf=conf, verbose=False)[0]
        lat.append((time.perf_counter() - t0) * 1000)
    return lat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imgs", default=str(REPO / "models" / "calib"))
    ap.add_argument("--n", type=int, default=60, help="对比采样帧数")
    ap.add_argument("--iters", type=int, default=30, help="性能测试次数")
    args = ap.parse_args()

    imgs = sorted(p for p in Path(args.imgs).iterdir()
                  if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"))
    if not imgs:
        raise SystemExit(f"无图片：{args.imgs}")
    imgs = imgs[: args.n]
    print(f"[verify] 采样 {len(imgs)} 张业务帧：{Path(args.imgs)}")

    m_fp16 = YOLO(str(FP16_DIR))
    m_int8 = YOLO(str(INT8_DIR))

    # --- 精度一致性 ---
    conf_deltas, ious, kp_dists = [], [], []
    n_det_fp16 = n_det_int8 = 0
    for p in imgs:
        rf = m_fp16(str(p), conf=0.3, verbose=False)[0]
        ri = m_int8(str(p), conf=0.3, verbose=False)[0]
        bf = rf.boxes.xyxy.numpy() if rf.boxes is not None else np.empty((0, 4))
        bi = ri.boxes.xyxy.numpy() if ri.boxes is not None else np.empty((0, 4))
        n_det_fp16 += len(bf)
        n_det_int8 += len(bi)
        pairs, miss_f, miss_i = _matched(bf, bi)
        for i, j, iou in pairs:
            ious.append(iou)
            conf_deltas.append(abs(float(rf.boxes.conf[i]) - float(ri.boxes.conf[j])))
        if rf.keypoints is not None and ri.keypoints is not None and pairs:
            kp_dists.append(_keypoint_dist(rf.keypoints.data[pairs[0][0]],
                                           ri.keypoints.data[pairs[0][1]]))

    print(f"[verify] 检测框：FP16 共 {n_det_fp16}，INT8 共 {n_det_int8}")
    if ious:
        print(f"[verify] 配对框 IoU  平均 {np.mean(ious):.4f}  最小 {np.min(ious):.4f}")
        print(f"[verify] 置信度差     平均 {np.mean(conf_deltas):.4f}  最大 {np.max(conf_deltas):.4f}")
    else:
        print("[verify] 无配对框，无法计算 IoU/置信度差")
    if kp_dists:
        print(f"[verify] 关键点平均距离 {np.mean(kp_dists):.2f} px")

    # --- 性能对比 ---
    print(f"\n[verify] 性能（单帧 640x640，CPU，预热后 {args.iters} 次）...")
    for name, m in (("FP16", m_fp16), ("INT8", m_int8)):
        lat = run(m, [str(p) for p in imgs], iters=args.iters)
        print(f"[verify] {name}: 平均 {np.mean(lat):.1f} ms  最大 {np.max(lat):.1f} ms")


if __name__ == "__main__":
    main()
