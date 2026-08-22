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
