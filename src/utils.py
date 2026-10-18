"""Shared helpers: ffmpeg/ffprobe invocation, hashing, logging, atomic writes."""
import io
import os
import subprocess
import tempfile
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


def atomic_write_text(path, content, encoding="utf-8"):
    """原子写文本：先写同目录临时文件并 fsync，再 os.replace 覆盖。

    进程崩溃/断电时目标文件要么是旧内容要么是新内容，不会出现半截文件
    （例如 config.json / tasks.json 写一半导致 JSON 解析失败）。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding=encoding, newline="") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def atomic_write_bytes(path, data):
    """原子写字节：先写同目录临时文件并 fsync，再 os.replace 覆盖。

    用于网格压缩图等二进制产物，避免写一半留下损坏文件。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def compress_jpeg_to_bytes(img, max_width, target_bytes,
                           min_quality=40, start_quality=85):
    """把 PIL 图片压缩成 JPEG 字节，尽量控制在 target_bytes 以内且保持清晰。

    机制：
    1. 若宽度超过 max_width，按宽度等比缩放到 max_width（高度随之等比，
       避免拉伸变形），从源头减小像素量。
    2. 从 start_quality 起逐步下调 JPEG 质量（每次 -5），直到输出体积
       <= target_bytes 或触达 min_quality 下限。质量有下限保证图不至于
       糊到看不清。
    3. 用 optimize + progressive 进一步减小体积。
    返回 (bytes, 实际采用的质量)。
    """
    if img.mode != "RGB":
        img = img.convert("RGB")
    if max_width and img.width > max_width:
        ratio = max_width / img.width
        new_size = (max_width, max(1, int(round(img.height * ratio))))
        img = img.resize(new_size, Image.LANCZOS)

    q = start_quality
    last = None
    while q >= min_quality:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=q, optimize=True,
                 progressive=True)
        size = buf.tell()
        last = (buf.getvalue(), q)
        if size <= target_bytes:
            return buf.getvalue(), q
        q -= 5
    # 触达下限仍超体积：返回下限质量版本（保证至少可用且尽量小）
    if last is not None:
        return last
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=min_quality, optimize=True,
             progressive=True)
    return buf.getvalue(), min_quality


def atomic_save_image(img, path, **save_kwargs):
    """原子保存 PIL 图片：先写同目录临时文件，再 os.replace 覆盖。

    避免快照/网格图写一半留下损坏文件（前端加载到坏图会空白）。
    img: PIL.Image 实例。
    格式：优先取调用方传入的 format；未传时按目标文件扩展名推断
    （临时文件是 .tmp 后缀，PIL 无法从扩展名推断格式，必须显式指定）。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fmt = save_kwargs.pop("format", None)
    if fmt is None:
        # 用 PIL 的扩展名注册表解析（.jpg→JPEG，.png→PNG 等），
        # 不能直接 .suffix.upper()（JPG 不是 SAVE 的合法键）。
        fmt = Image.registered_extensions().get(path.suffix.lower())
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    os.close(fd)
    try:
        img.save(tmp, format=fmt, **save_kwargs)
        with open(tmp, "rb") as f:
            pass  # 确保文件已完整关闭落盘
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
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
    clip_dur = end - start
    if clip_dur < 0.5:
        print(f"[clip] clip too short for {source.name} @ {center_sec:.1f}s")
        return False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # 重新编码（比 -c copy 更稳定，避免 hevc seek 后拷贝出错）。
    # 注意：-ss 在 -i 前为 input seek，seek 后输出时间戳从 0 开始，
    # 因此必须用 -t（持续时长）而非 -to（结束时间戳），否则时长会翻倍或为空。
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start:.2f}",
        "-i", str(source),
        "-t", f"{clip_dur:.2f}",
        # 缩到 1280 宽再重编码：4K 源直接转 H.264 重编码慢且文件巨大，
        # 大屏播放 1280p 足够，速度与体积均改善约 9 倍。
        "-vf", "scale=1280:-2",
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "aac",
        "-avoid_negative_ts", "make_zero",
        str(out_path),
    ]
    try:
        run(cmd, capture_output=True)
        print(f"[clip] saved {out_path.name} ({clip_dur:.1f}s from "
              f"{source.name}@{center_sec:.1f}s)")
        return True
    except Exception as e:
        print(f"[clip] failed for {source.name} @ {center_sec:.1f}s: {e}")
        # 清理可能产生的损坏文件，避免前端加载到坏 clip 后 fallback 到源视频
        try:
            if out_path.exists() and out_path.stat().st_size < 1024:
                out_path.unlink()
        except OSError:
            pass
        return False
