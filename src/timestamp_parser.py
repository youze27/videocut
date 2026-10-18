"""Filename timestamp parsers for surveillance camera footage.

Adapted from MorvanZhou/find-human-video. Each parser extracts a recording
datetime from a file path; used to decide "today" and clip continuity when
the filesystem mtime is unreliable (e.g. copied footage)."""
import re
from datetime import datetime
from pathlib import Path


class BaseTimestampParser:
    brand = "base"
    description = "base"

    def parse(self, file_path: Path):
        return None


_PARSERS = {}


def register_parser(name, cls, aliases=None):
    _PARSERS[name] = cls
    for a in (aliases or []):
        _PARSERS[a] = cls


class GenericParser(BaseTimestampParser):
    brand = "generic"
    description = "Generic: auto-detect common YYYYMMDD[_]HHMMSS patterns"

    _PATTERNS = [
        r"(\d{4})(\d{2})(\d{2})[_-]?(\d{2})(\d{2})(\d{2})",     # 20231115_143052
        r"(\d{4})-(\d{2})-(\d{2})[ _-](\d{2})[-:](\d{2})[-:](\d{2})",  # 2023-11-15 14-30-52
    ]

    def parse(self, file_path: Path):
        txt = f"{file_path.parent.name} {file_path.stem}"
        for pat in self._PATTERNS:
            m = re.search(pat, txt)
            if m:
                try:
                    return datetime(*(int(x) for x in m.groups()))
                except ValueError:
                    continue
        # Fallback: date in the parent folder, time in the file name, e.g.
        # 20231115/143052.mp4  ->  2023-11-15 14:30:52
        fd = re.search(r"(\d{4})(\d{2})(\d{2})", file_path.parent.name)
        ft = re.search(r"(\d{2})(\d{2})(\d{2})", file_path.stem)
        if fd and ft:
            try:
                return datetime(*(int(x) for x in fd.groups() + ft.groups()))
            except ValueError:
                pass
        return None


class XiaomiParser(BaseTimestampParser):
    brand = "xiaomi"
    description = ("Xiaomi: folder YYYYMMDDHH + file MM[mM]SS[sS], "
                   "or file with YYYYMMDDHHMMSS timestamps")

    def parse(self, file_path: Path):
        stem = file_path.stem

        # Format 2: 文件名内含完整时间戳，例如
        # 00_20260823200732_20260824065412.mp4（开始_结束）
        # 双时间戳时取最后一个（结束时间=视频录制完成时间），
        # 因为视频在结束时间才完整可用，用于 scan_process_hours 过滤更准确。
        matches = re.findall(
            r"(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})", stem)
        if matches:
            try:
                return datetime(*(int(x) for x in matches[-1]))
            except ValueError:
                pass

        # Format 1: 文件夹 YYYYMMDDHH + 文件 MMmSSs_*.mp4
        # 注意：小米摄像头文件名使用大写 M/S（如 02M56S），正则需大小写不敏感。
        fol = file_path.parent.name
        fm = re.match(r"(\d{4})(\d{2})(\d{2})(\d{2})", fol)
        if fm:
            y, mo, d, h = (int(x) for x in fm.groups())
            mm = re.match(r"(\d{2})[mM](\d{2})[sS]", stem)
            if mm:
                mi, s = (int(x) for x in mm.groups())
                try:
                    return datetime(y, mo, d, h, mi, s)
                except ValueError:
                    pass

        # Format 3: 文件名末尾含 Unix 时间戳，例如
        # 01M28S_1787040088.mp4（小米摄像头导出格式，无日期父文件夹时）。
        # 提取最后一个下划线后的纯数字，限定在 2000-2100 年范围内避免误识别。
        m = re.search(r"_(\d{10})(?:_|\b)", stem)
        if m:
            ts = int(m.group(1))
            try:
                dt = datetime.fromtimestamp(ts)
                if datetime(2000, 1, 1) <= dt <= datetime(2100, 1, 1):
                    return dt
            except (ValueError, OSError, OverflowError):
                pass

        return None


class HikvisionParser(BaseTimestampParser):
    brand = "hikvision"
    description = "Hikvision: YYYYMMDDHHMMSS in name or folder+file"

    def parse(self, file_path: Path):
        # Scan folder + file name together so "20231115/143052.mp4" also works.
        txt = f"{file_path.parent.name} {file_path.stem}"
        m = re.search(r"(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})", txt)
        if m:
            try:
                return datetime(*(int(x) for x in m.groups()))
            except ValueError:
                pass
        # Fallback: date in the parent folder, time in the file name.
        fd = re.search(r"(\d{4})(\d{2})(\d{2})", file_path.parent.name)
        ft = re.search(r"(\d{2})(\d{2})(\d{2})", file_path.stem)
        if fd and ft:
            try:
                return datetime(*(int(x) for x in fd.groups() + ft.groups()))
            except ValueError:
                pass
        return None


class DahuaParser(BaseTimestampParser):
    brand = "dahua"
    description = "Dahua: YYYY-MM-DD HH-MM-SS.mp4 or YYYYMMDD_HHMMSS.mp4"

    def parse(self, file_path: Path):
        txt = f"{file_path.parent.name} {file_path.stem}"
        m = re.search(
            r"(\d{4})[-_](\d{2})[-_](\d{2})[_ -](\d{2})[-:](\d{2})[-:](\d{2})",
            txt)
        if not m:
            return None
        try:
            return datetime(*(int(x) for x in m.groups()))
        except ValueError:
            return None


register_parser("generic", GenericParser)
register_parser("xiaomi", XiaomiParser)
register_parser("hikvision", HikvisionParser)
register_parser("dahua", DahuaParser)
register_parser("auto", GenericParser,
                aliases=["自动", "default"])


def parse_timestamp(file_path, brand="auto"):
    """Return a datetime from the filename, or None if it cannot be parsed.

    brand="auto" 时依次尝试所有已注册解析器（xiaomi → hikvision → dahua →
    generic），返回第一个成功的结果。这样混合摄像头品牌也能自动识别。"""
    file_path = Path(file_path)
    if brand == "auto":
        for name in ("xiaomi", "hikvision", "dahua", "generic"):
            cls = _PARSERS.get(name)
            if cls is None:
                continue
            dt = cls().parse(file_path)
            if dt is not None:
                return dt
        return None
    cls = _PARSERS.get(brand) or _PARSERS["generic"]
    return cls().parse(file_path)
