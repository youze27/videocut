"""Stitch deduped clips into a single compressed target-length video,
preserving each clip's original resolution (no scaling/cropping)."""
from pathlib import Path

import cv2
from . import config, utils


def _build_clip_list(kept_clips, target_duration):
    """Order clips and loop them if needed so total material >= target."""
    if not kept_clips:
        return []
    durations = [utils.video_duration(c) for c in kept_clips]
    total = sum(durations)

    clips = list(kept_clips)
    if total < target_duration and total > 0:
        i = 0
        while sum(utils.video_duration(c) for c in clips) < target_duration:
            clips.append(kept_clips[i % len(kept_clips)])
            i += 1
            if i > len(kept_clips) * 8:
                break
    return clips


def _resolution(path):
    cap = cv2.VideoCapture(str(path))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    return w, h


def stitch(kept_clips, output_path, target_duration=None):
    """Concatenate kept clips (native resolution), compress to
    `target_duration` seconds. A black placeholder is produced when no
    clips are kept so output always exists."""
    target_duration = target_duration or config.TARGET_DURATION
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    clips = _build_clip_list(kept_clips, target_duration)
    if not clips:
        print("[stitch] no clips kept -> producing black placeholder")
        cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i",
            f"color=c=black:s={config.OUT_WIDTH}x{config.OUT_HEIGHT}:d={target_duration}",
            "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
            "-c:v", "libx264", "-t", str(target_duration), "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", config.AUDIO_BITRATE,
            str(output_path),
        ]
        utils.run(cmd)
        return output_path

    # Preserve native resolution. If clips differ in size, ffmpeg concat
    # needs a uniform canvas, so scale to the first clip's resolution.
    base_w, base_h = _resolution(clips[0])
    need_scale = any(_resolution(c) != (base_w, base_h) for c in clips)
    if need_scale:
        print(f"[stitch] warning: resolutions differ; normalizing to "
              f"{base_w}x{base_h} (first clip's resolution)")

    n = len(clips)
    inputs = []
    for c in clips:
        inputs += ["-i", str(c)]

    vfilters, afilters = [], []
    for i in range(n):
        if need_scale:
            vfilters.append(f"[{i}:v]scale={base_w}:{base_h}[v{i}]")
        else:
            vfilters.append(f"[{i}:v]copy[v{i}]")
        if utils.has_audio(clips[i]):
            afilters.append(f"[{i}:a]aresample=44100[a{i}]")
        else:
            afilters.append(f"anullsrc=r=44100:cl=stereo:d=0.1[a{i}]")

    concat_v = "".join(f"[v{i}]" for i in range(n))
    concat_a = "".join(f"[a{i}]" for i in range(n))
    filter_complex = ";".join(vfilters + afilters) + \
        f";{concat_v}concat=n={n}:v=1:a=0[outv]" + \
        f";{concat_a}concat=n={n}:v=0:a=1[outa]"

    cmd = [
        "ffmpeg", "-y", *inputs,
        "-filter_complex", filter_complex,
        "-map", "[outv]", "-map", "[outa]",
        "-t", str(target_duration),
        "-c:v", "libx264", "-preset", config.PRESET, "-crf", config.CRF,
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", config.AUDIO_BITRATE,
        "-movflags", "+faststart",
        str(output_path),
    ]
    utils.run(cmd)
    return output_path
