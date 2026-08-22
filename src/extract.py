"""Extract frames from a video using FFmpeg (multi-threaded decode).

Decoding via FFmpeg is substantially faster than OpenCV's VideoCapture seek
and avoids keyframe seek overhead. Frames are written as JPGs to a temp dir
and returned as a sorted list of paths.
"""
import time
from pathlib import Path

from . import config, utils


def extract_frames(video_path, interval_sec=None, out_dir=None, max_frames=None):
    """Extract one frame every `interval_sec` seconds.

    Returns a sorted list of extracted JPG paths (may be empty)."""
    interval_sec = interval_sec if interval_sec is not None else config.DET_INTERVAL
    interval_sec = max(interval_sec, 0.1)
    out_dir = Path(out_dir) if out_dir else Path(config.WORK_DIR) / "frames"
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    # 清理上一轮残留帧
    old_count = 0
    for old in out_dir.glob("f_*.jpg"):
        try:
            old.unlink()
            old_count += 1
        except OSError:
            pass
    if old_count:
        print(f"[extract] cleaned {old_count} old frame(s) from {out_dir.name}")

    pattern = out_dir / "f_%06d.jpg"
    fps = 1.0 / interval_sec

    # 记录视频文件大小，帮助诊断大文件
    try:
        fsize = Path(video_path).stat().st_size
        fsize_str = f"{fsize / 1024 / 1024:.1f}MB" if fsize > 1024 * 1024 else f"{fsize / 1024:.0f}KB"
    except Exception:
        fsize_str = "?"
    print(f"[extract] video={Path(video_path).name} ({fsize_str}), "
          f"interval={interval_sec:.1f}s (fps={fps:.2f}), "
          f"out={out_dir.name}")

    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vf", f"fps={fps}", "-q:v", "3", str(pattern),
    ]
    try:
        utils.run(cmd)
    except Exception as e:
        print(f"[extract] ERROR: ffmpeg failed for {video_path}: {e}")
        raise

    frames = sorted(out_dir.glob("f_*.jpg"))
    if max_frames and len(frames) > max_frames:
        print(f"[extract] truncating {len(frames)} -> {max_frames} frames (limit)")
        frames = frames[:max_frames]

    elapsed = time.time() - t0
    print(f"[extract] extracted {len(frames)} frame(s) in {elapsed:.1f}s "
          f"({len(frames) / max(elapsed, 0.01):.0f} fps)")
    return frames


def extract_frames_at_times(video_path, times, out_dir):
    """在指定时间点各抽取一帧。

    times: [float, ...] 秒列表
    返回 [(time_sec, frame_path), ...] 成功抽取的帧。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for t in times:
        if t < 0:
            continue
        out_path = out_dir / f"t_{int(t*1000):08d}.jpg"
        cmd = [
            "ffmpeg", "-y", "-ss", f"{t:.2f}",
            "-i", str(video_path),
            "-frames:v", "1", "-q:v", "3", str(out_path),
        ]
        try:
            utils.run(cmd)
            if out_path.exists():
                results.append((t, out_path))
        except Exception as e:
            print(f"[extract] frame at {t:.1f}s failed: {e}")
    return results
