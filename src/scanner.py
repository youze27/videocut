"""Scan the input folder for video files whose date matches the target date.

Date is taken from the camera filename timestamp when `DATE_SOURCE` includes
"filename" (parsed via timestamp_parser), otherwise / as a fallback from the
filesystem mtime. "either" uses the filename timestamp if it parses, else mtime.
"""
from datetime import datetime, timedelta
from pathlib import Path

from . import config, utils
from .timestamp_parser import parse_timestamp


def target_date():
    return datetime.now().date() - timedelta(days=config.DATE_TOLERANCE_DAYS)


def _video_date(path, brand=None):
    """Return (datetime, source) for a video, or (None, None) if it cannot be
    dated under the current DATE_SOURCE setting.

    brand: 指定摄像头品牌用于文件名解析（覆盖 config.CAMERA_BRAND），
    这样不同目录可以用不同的品牌设置。"""
    camera_brand = brand or config.CAMERA_BRAND
    if config.DATE_SOURCE in ("filename", "either"):
        dt = parse_timestamp(path, camera_brand)
        if dt is not None:
            return dt, "filename"
    if config.DATE_SOURCE in ("mtime", "either"):
        return datetime.fromtimestamp(path.stat().st_mtime), "mtime"
    # filename-only but unparsed -> exclude
    return None, None


def video_datetime(path, brand=None):
    """Full recording datetime for a video (filename or mtime), or None."""
    dt, _ = _video_date(Path(path), brand=brand)
    return dt


def _list_subdirs(input_dir):
    """返回 input_dir 下所有子目录（按 mtime 倒序，最新在前）。"""
    try:
        subdirs = [d for d in input_dir.iterdir() if d.is_dir()]
    except OSError as e:
        print(f"[scan] WARNING: cannot list {input_dir}: {e}")
        return []
    subdirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    return subdirs


def _scan_one_dir(scan_dir, brand, dir_found, stats):
    """扫描单个目录下的视频文件，结果追加到 dir_found，统计追加到 stats。"""
    for p in sorted(scan_dir.iterdir()):
        if not p.is_file():
            continue
        if not utils.ffprobe_has_video(p):
            stats["nonvideo"] += 1
            continue
        vdt, date_src = _video_date(p, brand=brand)
        if vdt is None:
            stats["nodate"] += 1
            continue
        dir_found.append((p, vdt, date_src))
    stats["scanned"] += 1


def scan_videos(input_sources=None):
    """Return a list of video file paths to process.

    支持多个输入源（不同摄像头文件夹），每个源可独立指定摄像头品牌。
    自动识别目录结构：
      - 路径下有子目录（如日期文件夹 2026082121）→ 扫描全部子目录（不再只取最新）
      - 路径下无子目录（视频直接在当前目录）→ 直接扫描当前目录
    汇总所有视频后按时间倒序取最近 SCAN_LATEST_COUNT 个未处理视频。

    input_sources: list of {"path": str, "brand": str} 或 None（用 config）
    """
    if input_sources is None:
        input_sources = config.INPUT_SOURCES
        if not input_sources:
            input_sources = [{"path": str(config.INPUT_DIR), "brand": config.CAMERA_BRAND}]

    print(f"[scan] sources={len(input_sources)}")
    for s in input_sources:
        print(f"[scan]   dir={s.get('path')}, brand={s.get('brand', 'auto')}")

    found = []
    for src in input_sources:
        input_dir = Path(src.get("path", ""))
        brand = src.get("brand", "auto")
        if not input_dir.exists():
            print(f"[scan] WARNING: input dir does not exist: {input_dir}")
            continue

        # 扫描全部子目录（不再只取最新子目录），保证扫描完整
        subdirs = _list_subdirs(input_dir)
        scan_dirs = subdirs if subdirs else [input_dir]
        if subdirs:
            print(f"[scan] {input_dir}: found {len(subdirs)} subfolder(s), "
                  f"scanning ALL of them")
        else:
            print(f"[scan] {input_dir}: no subfolders, scanning directly")

        dir_found = []
        stats = {"scanned": 0, "nonvideo": 0, "nodate": 0}
        for sd in scan_dirs:
            _scan_one_dir(sd, brand, dir_found, stats)

        total_scanned = len(dir_found) + stats["nonvideo"] + stats["nodate"]
        print(f"[scan] {input_dir} (brand={brand}): scanned {stats['scanned']} "
              f"dir(s), {total_scanned} files, kept {len(dir_found)}, "
              f"skipped(non-video={stats['nonvideo']}, no-date={stats['nodate']})")

        if dir_found:
            # 汇总后按时间倒序，取最近 N 个
            dir_found.sort(key=lambda x: x[1], reverse=True)
            limit = getattr(config, "SCAN_LATEST_COUNT", 3)
            before = len(dir_found)
            dir_found = dir_found[:limit]
            if before > limit:
                print(f"[scan] {input_dir}: latest mode kept {limit}/{before} "
                      f"(dropped {before - limit} older)")
            # 再正序输出（处理顺序从旧到新）
            dir_found.sort(key=lambda x: x[1])

        for p, vdt, date_src in dir_found:
            found.append(p)
            print(f"[scan]   + {p.name} (date={vdt} via {date_src})")

    print(f"[scan] found {len(found)} video(s)")
    return found


def list_all_videos(input_dir=None, brand=None):
    """List every video file under input_dir with its parsed date (if any).

    与 scan_videos 同样的目录结构自动识别逻辑：有子目录则扫描全部子目录，
    无子目录则扫描当前目录。用于 web UI 扫描测试。"""
    input_dir = Path(input_dir or config.INPUT_DIR)
    if not input_dir.exists() or not input_dir.is_dir():
        return []

    # 扫描全部子目录（与 scan_videos 保持一致）
    subdirs = _list_subdirs(input_dir)
    scan_dirs = subdirs if subdirs else [input_dir]

    items = []
    for scan_dir in scan_dirs:
        for p in sorted(scan_dir.iterdir()):
            if not p.is_file():
                continue
            if not utils.ffprobe_has_video(p):
                continue
            vdt, src = _video_date(p, brand=brand)
            st = p.stat()
            items.append({
                "path": str(p),
                "name": p.name,
                "size": st.st_size,
                "mtime": datetime.fromtimestamp(st.st_mtime).isoformat(sep=" "),
                "date": vdt.isoformat(sep=" ") if vdt else None,
                "date_source": src,
            })
    return items
