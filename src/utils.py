"""Shared helpers: ffmpeg/ffprobe invocation, hashing, logging."""
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import imagehash


def run(cmd, **kwargs):
    """Run a command, streaming output, raising on failure."""
    print("+ " + " ".join(str(c) for c in cmd))
    try:
        return subprocess.run(cmd, check=True, **kwargs)
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode() if e.stderr else ""
        print(f"[utils] command failed (exit={e.returncode}): "
              f"{' '.join(str(c) for c in cmd)}")
        if stderr:
            print(f"[utils] stderr: {stderr[:500]}")
        raise


def ffprobe_has_video(path):
    """Return True if the file contains a video stream (i.e. is a video).

    优先用 ffprobe（更准确、更快）；ffprobe 不可用时回退到 cv2 验证。"""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=codec_type",
        "-of", "csv=p=0", str(path),
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return out.stdout.strip() != ""
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        pass
    # ffprobe 不可用 → cv2 回退
    try:
        import cv2
        cap = cv2.VideoCapture(str(path))
        ok = cap.isOpened()
        cap.release()
        return ok
    except Exception as e:
        print(f"[utils] ffprobe_has_video: both ffprobe and cv2 failed "
              f"for {path}: {e}")
        return False


def frame_hash(path, pos_frac=0.5):
    """Compute a perceptual hash of a frame.

    支持图片文件（.jpg/.png 等）和视频文件：
      - 图片：直接用 cv2.imread 读取
      - 视频：用 VideoCapture 按位置抽帧
    返回 imagehash.ImageHash 或 None。
    """
    p = str(path)
    # 图片文件直接读取
    lower = p.lower()
    if lower.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
        img = cv2.imread(p)
        if img is None:
            return None
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb).convert("L").resize((128, 128))
        return imagehash.phash(pil)
    # 视频文件用 VideoCapture
    cap = cv2.VideoCapture(p)
    if not cap.isOpened():
        cap.release()
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if total > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * pos_frac))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb).convert("L").resize((128, 128))
    return imagehash.phash(pil)


def hamm(a, b):
    if a is None or b is None:
        return 999
    return a - b  # ImageHash defines __sub__ as hamming distance


def video_duration(path):
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(path),
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(out.stdout.strip() or 0)
    except (subprocess.CalledProcessError, ValueError):
        return 0.0


def has_audio(path):
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "a:0",
        "-show_entries", "stream=index",
        "-of", "csv=p=0", str(path),
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError:
        return False
    return out.stdout.strip() != ""


def clip_video(source, center_sec, clip_seconds, out_path, duration=None):
    """从源视频裁剪 center_sec 前后 clip_seconds 秒的片段。

    source: 源视频路径
    center_sec: 事件在视频中的时间点（秒）
    clip_seconds: 前后各裁剪的秒数（总时长 = 2 * clip_seconds）
    out_path: 输出 mp4 路径
    duration: 源视频总时长（秒），None 时用 ffprobe 获取
    返回 True 成功 / False 失败。
    """
    source = Path(source)
    out_path = Path(out_path)
    if duration is None:
        duration = video_duration(source)
    if duration <= 0:
        print(f"[clip] cannot get duration for {source.name}")
        return False
    start = max(0, center_sec - clip_seconds)
    end = min(duration, center_sec + clip_seconds)
    if end - start < 0.5:
        print(f"[clip] clip too short for {source.name} @ {center_sec:.1f}s")
        return False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # 重新编码（比 -c copy 更稳定，避免 hevc seek 后拷贝出错）
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start:.2f}",
        "-to", f"{end:.2f}",
        "-i", str(source),
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "aac",
        "-avoid_negative_ts", "make_zero",
        str(out_path),
    ]
    try:
        run(cmd, capture_output=True)
        print(f"[clip] saved {out_path.name} ({end-start:.1f}s from {source.name})")
        return True
    except Exception as e:
        print(f"[clip] failed for {source.name} @ {center_sec:.1f}s: {e}")
        return False
