"""Build a vertical collage of the N frames with the most confident person
detections across today's videos. Full frames are kept (no crop/resize); they
are only padded with black bars to a common width for stacking. The top of the
image and the output filename both use the current date.

Frames are extracted via FFmpeg and scored with YOLOv8."""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

from . import config, scanner, utils
from . import extract
from .detect import detect_frames, detect_pose, draw_skeleton


def _date_str():
    return scanner.target_date().strftime("%Y-%m-%d")


def _load_font(size):
    candidates = [
        "DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "DejaVuSans.ttf",
    ]
    for c in candidates:
        try:
            return ImageFont.truetype(c, size)
        except Exception:
            continue
    return ImageFont.load_default()


def collect_best_frames(videos, samples=None, top_n=None, conf=None, pose=None):
    """Extract frames from each video, score by YOLOv8 confidence, and return
    the `top_n` frames (as PIL Images) with the highest person confidence."""
    if pose is None:
        pose = config.USE_POSE
    samples = samples if samples is not None else config.BEST_SAMPLES_PER_VIDEO
    top_n = top_n if top_n is not None else config.BEST_TOP_N

    scored = []  # (confidence, image_path, keypoints_or_None)
    for vid in videos:
        dur = utils.video_duration(vid)
        if dur <= 0:
            continue
        interval = max(dur / samples, 0.5)
        frames = extract.extract_frames(vid, interval=interval, max_frames=samples)
        if not frames:
            continue
        if pose:
            results = detect_pose([str(f) for f in frames], conf=conf)
        else:
            results = detect_frames([str(f) for f in frames], conf=conf)
        for f, r in zip(frames, results):
            if r is None:
                continue
            kp = r[1] if pose else None
            scored.append((r[0], f, kp))

    scored.sort(key=lambda t: t[0], reverse=True)
    images = []
    for _, p, kp in scored[:top_n]:
        im = Image.open(p).convert("RGB")
        if pose and kp is not None:
            draw_skeleton(im, kp)
        images.append(im)
    return images


def build_collage(frames, date_str, out_path, header_height=None):
    """Vertically stack `frames` (PIL images, padded to a common width) under a
    header band showing `date_str`. Returns the output path or None if empty."""
    if not frames:
        return None
    header_height = header_height or config.BEST_HEADER_HEIGHT

    max_w = max(im.width for im in frames)

    header = Image.new("RGB", (max_w, header_height), (0, 0, 0))
    d = ImageDraw.Draw(header)
    font = _load_font(max(24, header_height // 2))
    d.text((20, (header_height - font.size) // 2), date_str,
           fill=(255, 255, 255), font=font)

    padded = []
    for im in frames:
        if im.width < max_w:
            new = Image.new("RGB", (max_w, im.height), (0, 0, 0))
            new.paste(im, ((max_w - im.width) // 2, 0))
            padded.append(new)
        else:
            padded.append(im)

    total_h = header.height + sum(im.height for im in padded)
    canvas = Image.new("RGB", (max_w, total_h), (0, 0, 0))
    y = 0
    canvas.paste(header, (0, y))
    y += header.height
    for im in padded:
        canvas.paste(im, (0, y))
        y += im.height

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(str(out_path))
    return out_path


def make_best_frames_image(videos, output_dir, top_n=None, pose=None):
    """Produce the dated collage image from `videos`. Returns the path or
    None when no person frames were found."""
    frames = collect_best_frames(videos, top_n=top_n, pose=pose)
    if not frames:
        print("[bestframes] no person frames found -> skipping collage")
        return None
    date_str = _date_str()
    out_path = Path(output_dir) / f"{date_str}.jpg"
    result = build_collage(frames, date_str, out_path)
    if result:
        print(f"[bestframes] wrote {result} ({len(frames)} frames)")
    return result
