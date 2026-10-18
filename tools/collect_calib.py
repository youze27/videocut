"""Collect INT8 calibration images from real business videos.

从实际业务视频（test_video/，即流水线真实处理的监控视频）按固定间隔抽帧，
生成 NNCF 混合精度量化所需的校准集（models/calib/，建议 >=300 张）。
同时会把已有的业务快照（devtest/**/snapshots/*.jpg、devtest/work/frames/）一并
合并进来，最大化覆盖真实监控画面分布。

用法：
  python tools/collect_calib.py [--src-dir test_video] [--interval 0.5]
                                [--out models/calib] [--max-width 1280]

说明：校准图必须来自业务真实画面（本项目实际处理的视频），不要用无关图片，
否则 INT8 量化精度会严重下降。二次运行会清空 out 目录重新采集。
"""
import argparse
import shutil
import sys
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_SRC = REPO / "test_video"
DEFAULT_OUT = REPO / "models" / "calib"

VIDEO_EXTS = (".mp4", ".mov", ".mkv", ".avi", ".m4v", ".ts")
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")
# 额外的真实业务图片来源：既有快照与抽帧结果
EXTRA_DIRS = [
    REPO / "devtest" / "output" / "snapshots",
    REPO / "devtest" / "snapshots",
    REPO / "devtest" / "work" / "frames",
]


def _collect_images(out: Path):
    """把散落在 devtest 下的既有业务图片拷入 calib，避免重复抽取。"""
    copied = 0
    for d in EXTRA_DIRS:
        if not d.exists():
            continue
        for p in sorted(d.rglob("*")):
            if p.suffix.lower() not in IMAGE_EXTS:
                continue
            dst = out / f"extra_{uuid.uuid4().hex[:8]}{p.suffix.lower()}"
            try:
                shutil.copy(p, dst)
                copied += 1
            except OSError as e:
                print(f"[calib] 复制失败 {p}: {e}")
    return copied


def _extract_from_video(video: Path, interval: float, out: Path, max_width: int):
    """用 ffmpeg 按间隔抽帧（q:v=3 高质量），超出 max_width 时等比缩放。"""
    name = video.stem[:40]
    pattern = out / f"{name}_%06d.jpg"
    vf = f"fps={1.0 / interval}"
    if max_width:
        vf += f",scale='min({max_width},iw)':-2"
    cmd = [
        "ffmpeg", "-y", "-i", str(video),
        "-vf", vf, "-q:v", "3", str(pattern),
    ]
    import subprocess
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"[calib] ffmpeg 失败 {video.name}: {r.stderr[-300:]}")
        return 0
    return len(list(out.glob(f"{name}_*.jpg")))


def main():
    ap = argparse.ArgumentParser(description="Collect INT8 calibration images")
    ap.add_argument("--src-dir", default=str(DEFAULT_SRC),
                    help="业务视频目录（递归扫描所有视频）")
    ap.add_argument("--interval", type=float, default=0.5,
                    help="抽帧间隔秒数（越小帧越多，默认 0.5s）")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="校准图输出目录")
    ap.add_argument("--max-width", type=int, default=1280,
                    help="抽帧最大宽度（0=不缩放）")
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    # 1) 既有业务图片（快照 + 已抽帧）
    n_extra = _collect_images(out)

    # 2) 从业务视频抽帧
    src = Path(args.src_dir)
    if not src.exists():
        sys.exit(f"[calib] 源视频目录不存在：{src}")
    # 排除 clips 目录：其中的片段是从源视频裁剪的重复内容，抽帧只会造成重复，
    # 校准只需源视频即可覆盖真实监控画面分布。
    videos = [p for p in sorted(src.rglob("*"))
              if p.suffix.lower() in VIDEO_EXTS and "clips" not in p.parts]
    if not videos:
        sys.exit(f"[calib] {src} 下没有视频文件")
    n_frames = 0
    for v in videos:
        n_frames += _extract_from_video(v, args.interval, out, args.max_width)

    total = len(list(out.glob("*")))
    print(f"[calib] 完成：视频抽帧 {n_frames} 张 + 既有图片 {n_extra} 张 = "
          f"共 {total} 张校准图 -> {out}")
    if total < 300:
        print(f"[calib] 警告：校准图 <300，建议减小 --interval "
              f"(当前 {args.interval}s) 增加帧数。")


if __name__ == "__main__":
    main()
