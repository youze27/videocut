"""Scan the input folder for video files whose date matches the target date.

Date is taken from the camera filename timestamp when `DATE_SOURCE` includes
"filename" (parsed via timestamp_parser), otherwise / as a fallback from the
filesystem mtime. "either" uses the filename timestamp if it parses, else mtime.
"""
import re
from datetime import datetime, timedelta
from pathlib import Path

from . import config, utils
from .timestamp_parser import parse_timestamp

# 常见视频扩展名（小写）。用扩展名预过滤，避免对每个文件 fork ffprobe。
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".flv", ".wmv", ".m4v",
              ".mpg", ".mpeg", ".ts", ".m2ts", ".3gp", ".webm"}

# 文件夹名/文件名中的时间戳正则（用于快速判断是否当天，无需完整解析）
# 小时文件夹：YYYYMMDDHH（如 2026081621）
_RE_HOUR_FOLDER = re.compile(r"(\d{4})(\d{2})(\d{2})(\d{2})")
# 文件名内完整时间戳：YYYYMMDDHHMMSS（如 20260818062914）
_RE_FULL_TS = re.compile(r"(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})")
# 文件名末尾 Unix 时间戳：_NNNNNNNNNN（如 _1786887520）
_RE_UNIX_TS = re.compile(r"_(\d{10})(?:_|\b)")


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


def _quick_date(path):
    """快速从文件夹名/文件名提取日期，用于 smart 扫描预过滤当天视频。

    比 _video_date 快得多：只做正则匹配，不调用完整 timestamp_parser，
    不读 mtime（避免 stat 系统调用）。返回 datetime 或 None。

    支持的命名：
      - 小时文件夹 YYYYMMDDHH（如 2026081621）→ 取日期部分
      - 文件名完整时间戳 YYYYMMDDHHMMSS（如 20260818062914）
      - 文件名末尾 Unix 时间戳 _NNNNNNNNNN（如 _1786887520）
    """
    # 1. 文件名内完整时间戳（摄像头2: 00_20260818062914_20260818074636.mp4）
    stem = path.stem
    m = _RE_FULL_TS.search(stem)
    if m:
        try:
            return datetime(*(int(x) for x in m.groups()))
        except ValueError:
            pass
    # 2. 文件名末尾 Unix 时间戳（摄像头1: 38M40S_1786887520.mp4）
    m = _RE_UNIX_TS.search(stem)
    if m:
        try:
            ts = int(m.group(1))
            dt = datetime.fromtimestamp(ts)
            if datetime(2000, 1, 1) <= dt <= datetime(2100, 1, 1):
                return dt
        except (ValueError, OSError, OverflowError):
            pass
    # 3. 父文件夹名含 YYYYMMDDHH（摄像头1: 2026081621/）
    m = _RE_HOUR_FOLDER.search(path.parent.name)
    if m:
        try:
            return datetime(*(int(x) for x in m.groups()))
        except ValueError:
            pass
    return None


def _is_today(dt):
    """判断 datetime 是否在今天范围内（含 DATE_TOLERANCE_DAYS 容差）。"""
    if dt is None:
        return False
    today = datetime.now()
    tolerance = config.DATE_TOLERANCE_DAYS
    start = today.date() - timedelta(days=tolerance)
    return dt.date() >= start


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
        # 扩展名预过滤：非视频扩展名直接跳过，不调用 ffprobe（性能优化）
        if p.suffix.lower() not in VIDEO_EXTS:
            continue
        # 只扫当天：用快速日期预过滤当天视频（关闭则扫全部日期）
        if config.SCAN_TODAY_ONLY:
            qdt = _quick_date(p)
            if qdt is not None and not _is_today(qdt):
                stats["old_skipped"] += 1
                continue
            # qdt 为 None 时无法快速判断，保留待完整解析
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

    扫描模式（config.SCAN_MODE）：
      - "smart"（默认）：扩展名预过滤 + 文件夹/文件名快速判断当天，只扫当天视频，
        再过滤最近 SCAN_PROCESS_HOURS 小时的视频处理。扫描量最小。
      - "latest"：取最近 SCAN_LATEST_COUNT 个未处理视频。
      - "hours"：处理最近 SCAN_HOURS 小时的视频。

    input_sources: list of {"path": str, "brand": str} 或 None（用 config）
    """
    if input_sources is None:
        input_sources = config.INPUT_SOURCES
        if not input_sources:
            input_sources = [{"path": str(config.INPUT_DIR), "brand": config.CAMERA_BRAND}]

    print(f"[scan] sources={len(input_sources)}, mode={config.SCAN_MODE}")
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
        stats = {"scanned": 0, "nonvideo": 0, "nodate": 0, "old_skipped": 0}
        for sd in scan_dirs:
            _scan_one_dir(sd, brand, dir_found, stats)

        total_scanned = len(dir_found) + stats["nonvideo"] + stats["nodate"]
        skip_msg = (f", old_skipped={stats['old_skipped']}"
                    if stats["old_skipped"] else "")
        print(f"[scan] {input_dir} (brand={brand}): scanned {stats['scanned']} "
              f"dir(s), {total_scanned} files, kept {len(dir_found)}, "
              f"skipped(non-video={stats['nonvideo']}, no-date={stats['nodate']}"
              f"{skip_msg})")

        if dir_found:
            dir_found.sort(key=lambda x: x[1], reverse=True)  # 最新在前

            # 处理最近 N 小时：用户设几小时就只保留几小时内的视频（0 = 不限制）
            if config.SCAN_PROCESS_HOURS > 0:
                cutoff = datetime.now() - timedelta(hours=config.SCAN_PROCESS_HOURS)
                before = len(dir_found)
                dir_found = [x for x in dir_found if x[1] >= cutoff]
                if before > len(dir_found):
                    print(f"[scan] {input_dir}: 时间过滤保留 "
                          f"{len(dir_found)}/{before} (最近 {config.SCAN_PROCESS_HOURS}h, "
                          f"丢弃 {before - len(dir_found)} 个更早的)")

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
            # 扩展名预过滤（与 scan_videos 一致）
            if p.suffix.lower() not in VIDEO_EXTS:
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
