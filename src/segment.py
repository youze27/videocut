"""Split a source video into fixed-length candidate segments."""
from pathlib import Path

from . import config, utils


def split_segments(video_path, work_dir, seg_id_prefix, segment_duration=None):
    """Split `video_path` into ~segment_duration second clips.

    Returns a list of (segment_path, start_time, end_time) tuples.
    Uses ffmpeg's segment muxer so streams (incl. audio) are preserved."""
    segment_duration = segment_duration or config.SEGMENT_DURATION
    seg_dir = Path(work_dir) / "segments" / seg_id_prefix
    seg_dir.mkdir(parents=True, exist_ok=True)
    pattern = seg_dir / f"{seg_id_prefix}_%03d.mp4"

    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-map", "0", "-c", "copy",
        "-f", "segment", "-segment_time", str(segment_duration),
        "-reset_timestamps", "1", str(pattern),
    ]
    utils.run(cmd)

    segments = []
    for seg in sorted(seg_dir.glob(f"{seg_id_prefix}_*.mp4")):
        start = _segment_start(seg, seg_id_prefix, segment_duration)
        duration = utils.video_duration(seg)
        segments.append((seg, start, start + duration))
    return segments


def _segment_start(seg_path, prefix, segment_duration=None):
    """Infer segment start time from its numeric suffix."""
    segment_duration = segment_duration or config.SEGMENT_DURATION
    name = seg_path.stem
    suffix = name[len(prefix) + 1:]  # after the underscore
    try:
        idx = int(suffix)
    except ValueError:
        return 0.0
    return idx * segment_duration
