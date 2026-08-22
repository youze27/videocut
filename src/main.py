"""videocut: scan videos, extract frames, detect person/pet, and save a
cropped 3:2 JPG snapshot (with pose label) for each detected event.

本项目不再生成合并视频或拼图。每个检测到人/猫/狗的片段会输出一张围绕
检测框裁剪的 JPG 快照，并在 SNAPSHOTS_DIR/index.json 中记录源视频路径
与起始时间，大屏点击图片时直接跳转到源视频的对应位置播放。"""
import argparse
import json
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

from PIL import Image

from . import config
from . import scanner
from . import extract
from .extract import extract_frames, extract_frames_at_times
from . import detect
from . import dedup
from . import utils
from .detect import draw_skeleton


def _processed_path():
    return Path(config.WORK_DIR) / "processed.json"


def _no_obj_path():
    return Path(config.WORK_DIR) / "no_object_videos.json"


def load_no_obj():
    p = _no_obj_path()
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return []


def save_no_obj(data):
    p = _no_obj_path()
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False),
                 encoding="utf-8")


def load_processed():
    """已处理视频记录：{key: iso_timestamp}。key = path|size|mtime 保证文件
    被替换后能重新处理。"""
    p = _processed_path()
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def save_processed(proc):
    p = _processed_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(proc, indent=2, ensure_ascii=False), encoding="utf-8")


def _processed_key(path):
    """Stable key for a video file: path + size + mtime."""
    st = Path(path).stat()
    return f"{path}|{st.st_size}|{int(st.st_mtime)}"


def _events_index_path():
    return Path(config.SNAPSHOTS_DIR) / "index.json"


def _record_event(base, meta):
    """把每张事件 JPG 对应的源视频信息写入 index.json，供大屏跳转播放。"""
    p = _events_index_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    index = {}
    if p.exists():
        try:
            index = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    index[base] = meta
    p.write_text(json.dumps(index, indent=2, ensure_ascii=False),
                 encoding="utf-8")


def _dynamic_interval(duration):
    """根据视频时长动态计算抽帧间隔，平衡检测覆盖度与处理速度。

    小视频（<2分钟，如每小时1分钟的摄像头片段）：用配置的 DET_INTERVAL（默认2s）
    中等视频（2-10分钟）：至少 5s
    大视频（>10分钟，如128M的连续录像）：至少 10s
    """
    if duration > 600:
        return max(config.DET_INTERVAL, 10.0)
    elif duration > 120:
        return max(config.DET_INTERVAL, 5.0)
    return config.DET_INTERVAL


def _resolve_worker_threads():
    """解析 WORKER_THREADS 配置，返回实际线程数。

    "auto" → 自动按 CPU 核心数（最多 4，最少 1）
    整数字符串 → 该数字（至少 1）
    """
    raw = str(config.WORKER_THREADS).strip().lower()
    if raw == "auto":
        cores = os.cpu_count() or 1
        return max(1, min(4, cores))
    try:
        n = int(raw)
        return max(1, n)
    except (ValueError, TypeError):
        return 1


def cleanup_old_frames(work_dir=None, retention_days=None):
    """清理超过保留天数的临时帧文件（WORK_DIR/frames 下）。

    已输出的快照（SNAPSHOTS_DIR 下）不会被删除。
    retention_days=0 表示不清理。
    """
    work_dir = Path(work_dir or config.WORK_DIR)
    retention_days = retention_days if retention_days is not None else config.FRAME_RETENTION_DAYS
    if retention_days <= 0:
        return {"deleted": 0, "skipped": "retention=0"}
    frames_dir = work_dir / "frames"
    if not frames_dir.exists():
        return {"deleted": 0, "skipped": "no frames dir"}
    cutoff = time.time() - retention_days * 86400
    deleted = 0
    errors = 0
    for p in frames_dir.rglob("*.jpg"):
        try:
            if p.stat().st_mtime < cutoff:
                p.unlink()
                deleted += 1
        except OSError:
            errors += 1
    msg = f"cleanup: deleted {deleted} frame(s) older than {retention_days}d"
    if errors:
        msg += f" ({errors} errors)"
    print(f"[main] {msg}")
    return {"deleted": deleted, "errors": errors}


def _process_one_video(vid, vid_idx, vid_total, work_dir, processed, dfilter,
                       brand_map, cancel_flag, progress_cb, lock):
    """处理单个视频：抽帧→检测→快照。线程安全（dfilter/processed/no_obj 加锁）。

    返回 (key, snapshot_count, summary_dict)。
    """
    def _cancelled():
        return bool(cancel_flag and cancel_flag.get("cancel"))

    def _progress(stage, text, current=0, total=0, extra=None):
        if progress_cb:
            try:
                info = {"stage": stage, "text": text,
                        "current": current, "total": total}
                if extra:
                    info["extra"] = extra
                progress_cb(info)
            except Exception:
                pass

    key = _processed_key(vid)
    if key in processed:
        print(f"[main] [{vid_idx}/{vid_total}] SKIP {vid.name}: already processed")
        _progress("video_done", f"跳过已处理 [{vid_idx}/{vid_total}] {vid.name}",
                  vid_idx, vid_total, extra={
            "video": vid.name, "skipped": True, "duration": 0,
            "frames_extracted": 0, "frames_detected": 0, "snapshots": 0,
            "dedup_skipped": 0, "no_detection": 0, "elapsed": 0, "date": "",
        })
        return key, 0, None

    vid_t0 = time.time()
    vid_brand = _brand_for(vid, brand_map)
    vdt = scanner.video_datetime(vid, brand=vid_brand)
    try:
        dur = utils.video_duration(vid)
    except Exception:
        dur = 0
    interval = _dynamic_interval(dur)
    expected_frames = int(dur / interval) if dur > 0 else 0
    _progress("extract", f"抽取帧 [{vid_idx}/{vid_total}] {vid.name}",
              vid_idx, vid_total)
    print(f"[main] [{vid_idx}/{vid_total}] processing {vid.name}")
    print(f"[main]   path={vid}")
    print(f"[main]   duration={dur:.1f}s, interval={interval:.1f}s, "
          f"expected_frames~{expected_frames}, date={vdt}")

    frames = extract.extract_frames(vid, interval_sec=interval,
                                    out_dir=work_dir / "frames" / vid.stem,
                                    max_frames=120)
    if not frames:
        print(f"[main]   WARNING: no frames extracted, skipping")
        return key, 0, None

    _progress("detect", f"检测物体 [{vid_idx}/{vid_total}] {vid.name} "
              f"({len(frames)}帧)", vid_idx, vid_total)
    dets = detect.detect_objects([str(f) for f in frames],
                                 person_conf=config.DET_CONF,
                                 pet_conf=config.PET_CONF,
                                 pose=config.USE_POSE)

    kept_in_video = 0
    dedup_skipped = 0
    for idx, (frame_path, det) in enumerate(zip(frames, dets)):
        if _cancelled():
            print("[main] cancelled by user")
            break
        if det is None:
            continue
        frame_time = idx * interval
        frame_hash = utils.frame_hash(frame_path)

        with lock:
            dup, reason = dfilter.is_duplicate(
                vid, frame_time - interval / 2,
                frame_time + interval / 2, frame_hash)
            if dup:
                dedup_skipped += 1
                continue
            dfilter.add(vid, frame_time - interval / 2,
                        frame_time + interval / 2, frame_hash)

        analyze = dict(det)
        analyze["frame_path"] = str(frame_path)

        if (analyze["obj_type"] == "person" and
                analyze.get("pose") == "unknown" and
                config.USE_POSE and dur > 0):
            analyze = _retry_pose(
                vid, frame_time, dur, analyze,
                work_dir / "frames" / vid.stem / "retry",
                vid_idx, vid_total, _progress)

        print(f"[main]   keep frame {idx} @ {frame_time:.1f}s "
              f"({analyze['obj_type']}/{analyze['pose']}, "
              f"score={analyze['score']:.2f}, bbox={analyze['bbox']})")

        emit_frame_path = analyze.get("frame_path", str(frame_path))
        emit_time = analyze.pop("_retry_time", None) or frame_time
        etime = (vdt or datetime.now()) + timedelta(seconds=emit_time)
        _emit_event(emit_frame_path, analyze, etime,
                    source_video=vid, start_time=emit_time)
        kept_in_video += 1
        _progress("snapshot", f"保存快照 [{vid_idx}/{vid_total}] {vid.name} "
                  f"第{kept_in_video}张", vid_idx, vid_total)

    vid_elapsed = time.time() - vid_t0
    no_detection = len(frames) - kept_in_video - dedup_skipped
    print(f"[main]   {vid.name}: {kept_in_video} snapshot(s) saved, "
          f"{dedup_skipped} dedup-skipped, "
          f"{no_detection} no-detection, "
          f"took {vid_elapsed:.1f}s")
    _progress("video_done", f"完成 {vid.name}", vid_idx, vid_total, extra={
        "video": vid.name,
        "duration": round(dur, 1),
        "frames_extracted": len(frames),
        "frames_detected": kept_in_video + dedup_skipped,
        "snapshots": kept_in_video,
        "dedup_skipped": dedup_skipped,
        "no_detection": no_detection,
        "elapsed": round(vid_elapsed, 1),
        "date": str(vdt) if vdt else "",
    })

    # 记录无对象视频（线程安全）
    total_detected = kept_in_video + dedup_skipped
    with lock:
        no_obj = load_no_obj()
        if total_detected == 0:
            if not any(v.get("path") == str(vid) for v in no_obj):
                no_obj.append({
                    "path": str(vid), "name": vid.name,
                    "duration": f"{dur:.1f}s",
                    "date": str(vdt) if vdt else "",
                })
                save_no_obj(no_obj)
                print(f"[main]   recorded as no-object video")
        else:
            filtered = [v for v in no_obj if v.get("path") != str(vid)]
            if len(filtered) != len(no_obj):
                save_no_obj(filtered)

    return key, kept_in_video, None


def _brand_for(vid_path, brand_map):
    """根据视频路径在 brand_map 中查找对应的摄像头品牌。"""
    try:
        p = Path(vid_path).resolve()
        for base, brand in brand_map.items():
            try:
                p.relative_to(base)
                return brand
            except ValueError:
                continue
    except Exception:
        pass
    return config.CAMERA_BRAND


def process(work_dir, cancel_flag=None, progress_cb=None):
    work_dir = Path(work_dir)
    (work_dir / "frames").mkdir(parents=True, exist_ok=True)
    config.SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)

    def _cancelled():
        return bool(cancel_flag and cancel_flag.get("cancel"))

    def _progress(stage, text, current=0, total=0, extra=None):
        if progress_cb:
            try:
                info = {"stage": stage, "text": text,
                        "current": current, "total": total}
                if extra:
                    info["extra"] = extra
                progress_cb(info)
            except Exception:
                pass

    # 清理过期帧
    cleanup_old_frames(work_dir)

    _progress("scan", "正在扫描视频文件...", 0, 0)
    videos = scanner.scan_videos()
    if not videos:
        print("[main] no videos found for this run; finishing.")
        _progress("done", "未找到视频，结束", 0, 0)
        return {"snapshots": 0}

    processed = load_processed()
    print(f"[main] processed.json: {len(processed)} previously processed video(s)")
    dfilter = dedup.DedupFilter()
    snapshot_count = 0
    newly_processed = []
    pipeline_t0 = time.time()

    # 构建 path→brand 映射
    brand_map = {}
    for src in config.INPUT_SOURCES:
        try:
            base = Path(src.get("path", "")).resolve()
            brand_map[base] = src.get("brand", "auto")
        except Exception:
            pass

    import threading
    lock = threading.Lock()
    vid_total = len(videos)

    # 决定线程数
    worker_threads = _resolve_worker_threads()
    print(f"[main] worker_threads={worker_threads} (cpu_count={os.cpu_count()})")
    _progress("scan", f"共 {vid_total} 个视频，{worker_threads} 线程并行处理",
              0, vid_total)

    if worker_threads <= 1 or vid_total <= 1:
        # 单线程模式（保持原有顺序，调试方便）
        for vid_idx, vid in enumerate(videos, 1):
            if _cancelled():
                print("[main] cancelled by user")
                break
            key, snaps, _ = _process_one_video(
                vid, vid_idx, vid_total, work_dir, processed, dfilter,
                brand_map, cancel_flag, _progress, lock)
            snapshot_count += snaps
            if key:
                newly_processed.append(key)
    else:
        # 多线程模式：用 ThreadPoolExecutor 并行处理多个视频
        with ThreadPoolExecutor(max_workers=worker_threads) as pool:
            future_to_vid = {}
            for vid_idx, vid in enumerate(videos, 1):
                fut = pool.submit(_process_one_video, vid, vid_idx, vid_total,
                                  work_dir, processed, dfilter, brand_map,
                                  cancel_flag, _progress, lock)
                future_to_vid[fut] = vid
            for fut in as_completed(future_to_vid):
                if _cancelled():
                    # 取消未开始的任务
                    for f in future_to_vid:
                        f.cancel()
                    break
                try:
                    key, snaps, _ = fut.result()
                    snapshot_count += snaps
                    if key:
                        newly_processed.append(key)
                except Exception as e:
                    vid = future_to_vid[fut]
                    print(f"[main] ERROR processing {vid.name}: {e}")

    total_elapsed = time.time() - pipeline_t0
    print(f"[main] pipeline finished: {snapshot_count} snapshot(s) from "
          f"{len(videos)} video(s) in {total_elapsed:.1f}s")
    _progress("done", f"完成：{snapshot_count} 张快照，{len(videos)} 个视频，"
              f"用时 {total_elapsed:.1f}s", len(videos), len(videos))

    # 持久化已处理记录
    ts = datetime.now().isoformat()
    for key in newly_processed:
        processed[key] = ts
    if newly_processed:
        save_processed(processed)
        print(f"[main] marked {len(newly_processed)} video(s) as processed")

    return {"snapshots": snapshot_count}


def _retry_pose(vid, frame_time, dur, orig_analyze, retry_dir,
                vid_idx, vid_total, progress_cb=None):
    """姿态识别失败时，在附近时间点重新抽帧二次检测。

    策略：在 frame_time ±6s 范围内，每 2s 抽一帧，重新检测。
    如果任何一帧能识别出姿态（非 unknown），用该结果替换原结果。
    全部失败则返回原结果。
    """
    try:
        # 生成附近时间点：±2s, ±4s, ±6s
        offsets = [-6, -4, -2, 2, 4, 6]
        times = []
        for off in offsets:
            t = frame_time + off
            if 0 <= t <= dur:
                times.append(t)
        if not times:
            return orig_analyze

        if progress_cb:
            progress_cb({"stage": "detect",
                         "text": f"姿态重试 [{vid_idx}/{vid_total}] {vid.name} @ {frame_time:.0f}s",
                         "current": vid_idx, "total": vid_total})
        print(f"[main]   pose retry: extracting {len(times)} frames around {frame_time:.1f}s")

        retry_frames = extract_frames_at_times(vid, times, retry_dir)
        if not retry_frames:
            print(f"[main]   pose retry: no frames extracted, keeping original")
            return orig_analyze

        # 对重试帧运行检测
        frame_paths = [str(fp) for _, fp in retry_frames]
        retry_dets = detect.detect_objects(frame_paths,
                                           person_conf=config.DET_CONF,
                                           pet_conf=config.PET_CONF,
                                           pose=config.USE_POSE)

        # 找第一个姿态非 unknown 的人物检测结果
        for (rt, rfp), rdet in zip(retry_frames, retry_dets):
            if rdet is None:
                continue
            if rdet["obj_type"] == "person" and rdet.get("pose") != "unknown":
                print(f"[main]   pose retry SUCCESS: found '{rdet['pose']}' at {rt:.1f}s "
                      f"(score={rdet['score']:.2f})")
                result = dict(rdet)
                result["frame_path"] = str(rfp)
                # 更新 frame_time 为重试成功的时间点
                result["_retry_time"] = rt
                return result

        print(f"[main]   pose retry: all {len(retry_frames)} frames still unknown, keeping original")
    except Exception as e:
        print(f"[main]   pose retry ERROR: {e}")
    return orig_analyze


def _crop_to_subject(im, bbox, ratio=(3, 2), padding=0.25):
    """围绕检测框裁剪，统一为指定宽高比（默认 3:2 横向），突出主体。

    padding 控制裁剪区域相对检测框较大边的扩展比例（0.25 = 扩 25%）。
    主体靠近画面边缘、平移后仍不满足比例时，用黑边补齐，保证所有输出
    宽高比完全一致。"""
    W, H = im.size
    rw, rh = ratio

    if bbox:
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        bw = max(x2 - x1, 1)
        bh = max(y2 - y1, 1)
        base = max(bw, bh) * (1 + 2 * padding)
    else:
        cx, cy = W / 2, H / 2
        base = min(W, H)

    # 按目标比例计算裁剪宽高（以 base 为基准边）
    if rw >= rh:          # 横向比例
        crop_h = base
        crop_w = base * rw / rh
    else:                 # 纵向比例
        crop_w = base
        crop_h = base * rh / rw

    # 居中裁剪框
    left = cx - crop_w / 2
    top = cy - crop_h / 2
    right = cx + crop_w / 2
    bottom = cy + crop_h / 2

    # 平移到画面内（优先保持主体居中）
    if left < 0:
        right -= left; left = 0
    if top < 0:
        bottom -= top; top = 0
    if right > W:
        left -= (right - W); right = W
    if bottom > H:
        top -= (bottom - H); bottom = H
    left = max(0, left); top = max(0, top)
    right = min(W, right); bottom = min(H, bottom)

    cropped = im.crop((int(left), int(top), int(right), int(bottom)))
    cw, ch = cropped.size

    # 若主体贴边导致比例不符，用黑边补齐到精确比例
    target_ratio = rw / rh
    if abs((cw / ch) - target_ratio) > 0.01:
        if cw / ch < target_ratio:
            # 需要更宽：左右补黑边
            new_w = int(round(ch * target_ratio))
            canvas = Image.new("RGB", (new_w, ch), (0, 0, 0))
            canvas.paste(cropped, ((new_w - cw) // 2, 0))
        else:
            # 需要更高：上下补黑边
            new_h = int(round(cw / target_ratio))
            canvas = Image.new("RGB", (cw, new_h), (0, 0, 0))
            canvas.paste(cropped, (0, (new_h - ch) // 2))
        cropped = canvas

    return cropped


def _emit_event(seg_path, analyze, event_time, source_video=None,
                start_time=0.0):
    """只写一张 JPG 关键帧（人物绘制骨架），并在 index.json 中记录源视频
    路径与起始时间，大屏据此跳转到源视频播放。不再复制裁剪后的 MP4。

    快照会围绕检测框裁剪到统一 3:2 比例并缩放到固定尺寸，突出主体，
    大屏展示更整齐美观。

    文件命名： <YYYYMMDD_HHMMSS>_<obj>_pose-<pose>_<rand>.jpg"""
    ts = event_time.strftime("%Y%m%d_%H%M%S")
    rand = uuid.uuid4().hex[:3]
    obj = analyze["obj_type"]
    pose = analyze["pose"]
    base = f"{ts}_{obj}_pose-{pose}_{rand}"

    snap = config.SNAPSHOTS_DIR / f"{base}.jpg"
    im = Image.open(analyze["frame_path"]).convert("RGB")
    # 先在原图上画骨架（关键点坐标是原图坐标），再裁剪
    if obj == "person" and analyze.get("keypoints") is not None:
        draw_skeleton(im, analyze["keypoints"], analyze.get("kconf"))
    # 围绕检测框裁剪到统一比例，突出主体
    im = _crop_to_subject(im, analyze.get("bbox"),
                          ratio=config.SNAPSHOT_RATIO,
                          padding=config.SNAPSHOT_PADDING)
    # 统一缩放到固定尺寸
    tw = config.SNAPSHOT_WIDTH
    th = int(tw * config.SNAPSHOT_RATIO[1] / config.SNAPSHOT_RATIO[0])
    im = im.resize((tw, th), Image.Resampling.LANCZOS)
    im.save(str(snap), quality=92)

    _record_event(base, {
        "source_video": str(source_video) if source_video else None,
        "start_time": float(start_time),
        "obj": obj,
        "pose": pose,
        "snapshot": snap.name,
    })
    print(f"[events] saved {snap.name} ({tw}x{th}, "
          f"src={Path(source_video).name if source_video else 'N/A'}, "
          f"t={start_time:.1f}s)")


def run_pipeline(input_dir=None, output_dir=None, work_dir=None,
                 det_conf=None,
                 similarity_threshold=None, det_interval=None,
                 date_source=None, camera_brand=None, use_pose=None,
                 scan_hours=None, scan_mode=None, cancel_flag=None,
                 input_sources=None, progress_cb=None,
                 worker_threads=None, frame_retention_days=None):
    """Configure and run the scan + detect + snapshot pipeline. All arguments
    are optional and fall back to the current config values. Returns a dict
    with the count of snapshots saved.

    cancel_flag: 可选的 dict，如果其 'cancel' 键为 True 则提前退出循环。
    input_sources: 可选的列表 [{"path": str, "brand": str}, ...]，
    指定时覆盖 input_dir/camera_brand 的单目录配置。
    worker_threads: "auto" 或整数，控制多视频并行处理的线程数。
    frame_retention_days: 临时帧保留天数，超过自动删除。"""
    if input_sources is not None:
        config.INPUT_SOURCES = input_sources
        if input_sources:
            config.INPUT_DIR = Path(input_sources[0].get("path", "/input"))
    elif input_dir is not None:
        # input_dir 可以是逗号分隔的多个目录（兼容旧接口），brand 用 camera_brand
        dirs = [d.strip() for d in str(input_dir).split(",") if d.strip()]
        config.INPUT_SOURCES = [{"path": d, "brand": camera_brand or "auto"}
                                 for d in dirs]
        config.INPUT_DIR = Path(dirs[0]) if dirs else Path(input_dir)
    if output_dir is not None:
        config.OUTPUT_DIR = Path(output_dir)
        # 事件快照目录随输出目录同步
        config.SNAPSHOTS_DIR = config.OUTPUT_DIR / "snapshots"
    if work_dir is not None:
        config.WORK_DIR = Path(work_dir)
    if det_conf is not None:
        config.DET_CONF = det_conf
    if similarity_threshold is not None:
        config.SIMILARITY_THRESHOLD = similarity_threshold
    if det_interval is not None:
        config.DET_INTERVAL = float(det_interval)
    if date_source is not None:
        config.DATE_SOURCE = date_source
    if camera_brand is not None:
        config.CAMERA_BRAND = camera_brand
    if use_pose is not None:
        config.USE_POSE = (str(use_pose).lower() == "true")
    if scan_hours is not None:
        try:
            config.SCAN_HOURS = float(scan_hours)
        except (TypeError, ValueError):
            pass
    if scan_mode is not None:
        config.SCAN_MODE = scan_mode
    if worker_threads is not None:
        config.WORKER_THREADS = worker_threads
    if frame_retention_days is not None:
        try:
            config.FRAME_RETENTION_DAYS = int(frame_retention_days)
        except (TypeError, ValueError):
            pass

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return process(config.WORK_DIR, cancel_flag=cancel_flag, progress_cb=progress_cb)


def main():
    ap = argparse.ArgumentParser(description="videocut snapshot tool")
    ap.add_argument("--input", default=None, help="input folder (overrides INPUT_DIR)")
    ap.add_argument("--output", default=None, help="output folder (overrides OUTPUT_DIR)")
    ap.add_argument("--work", default=None, help="scratch folder (overrides WORK_DIR)")
    args = ap.parse_args()

    run_pipeline(
        input_dir=args.input, output_dir=args.output, work_dir=args.work,
    )


if __name__ == "__main__":
    main()
