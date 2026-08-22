"""Decide whether a clip contains a person/pet and classify it.

Frames are extracted with FFmpeg (fast, multi-threaded). YOLOv8 detects
person/cat/dog and (for persons) YOLOv8-pose keypoints drive a pose label.
No frame resizing/cropping happens -- the original frames are kept as-is."""
from pathlib import Path

from . import config
from . import extract
from .detect import detect_frames, detect_objects


def segment_has_person(video_path, interval=None, conf=None, pose=None):
    """Extract frames at `interval` seconds; return True if a person is
    detected in any of them."""
    if pose is None:
        pose = config.USE_POSE
    frames = extract.extract_frames(video_path, interval, max_frames=64)
    if not frames:
        return False
    results = detect_frames([str(f) for f in frames], conf=conf, pose=pose)
    return any(r is not None for r in results)


def analyze_segment(seg_path, interval=None, person_conf=None, pet_conf=None,
                    pose=None):
    """Extract frames and detect the dominant object (person/cat/dog) plus its
    pose. Returns a dict {obj_type, score, pose, keypoints, kconf, bbox,
    frame_path} for the best frame, or None when nothing is detected."""
    if pose is None:
        pose = config.USE_POSE
    frames = extract.extract_frames(seg_path, interval, max_frames=24)
    if not frames:
        return None
    dets = detect_objects([str(f) for f in frames], person_conf=person_conf,
                          pet_conf=pet_conf, pose=pose)
    best = None  # (score, idx, det)
    for idx, (f, det) in enumerate(zip(frames, dets)):
        if det is None:
            continue
        if best is None or det["score"] > best[0]:
            best = (det["score"], idx, det)
    if best is None:
        return None
    score, idx, det = best
    det = dict(det)
    det["frame_path"] = str(frames[idx])
    return det
