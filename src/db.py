"""SQLite persistence for processed videos, no-object videos and event metadata.

替代旧 JSON 文件存储（processed.json / no_object_videos.json / index.json），
提供线程安全的单连接访问，并在首次初始化时一次性迁移存量 JSON 数据。

DB 位置：<WORK_DIR>/videocut.db
表：
  processed_videos(key TEXT PRIMARY KEY, processed_at TEXT)
  no_object_videos(path TEXT PRIMARY KEY, name TEXT, duration TEXT, date TEXT)
  events(base TEXT PRIMARY KEY, source_video TEXT, start_time REAL,
         obj TEXT, pose TEXT, snapshot TEXT, clip TEXT)
  task_runs(task_id TEXT PRIMARY KEY, started, finished, status, ...) 每次任务运行汇总
  video_runs(id, task_id, video, ...) 每次任务下每个视频的处理明细
"""
import json
import sqlite3
import threading
from pathlib import Path

from . import config

_conn = None
_lock = threading.Lock()


def _db_path():
    return Path(config.WORK_DIR) / "videocut.db"


def _get_conn():
    global _conn
    if _conn is None:
        with _lock:
            if _conn is None:
                p = _db_path()
                p.parent.mkdir(parents=True, exist_ok=True)
                _conn = sqlite3.connect(str(p), check_same_thread=False)
                _conn.execute("PRAGMA journal_mode=WAL")
                _create_tables()
                _migrate_from_json_once()
    return _conn


def _create_tables():
    _conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS processed_videos (
            key          TEXT PRIMARY KEY,
            processed_at TEXT
        );
        CREATE TABLE IF NOT EXISTS no_object_videos (
            path     TEXT PRIMARY KEY,
            name     TEXT,
            duration TEXT,
            date     TEXT
        );
        CREATE TABLE IF NOT EXISTS events (
            base         TEXT PRIMARY KEY,
            source_video TEXT,
            start_time   REAL,
            obj          TEXT,
            pose         TEXT,
            snapshot     TEXT,
            clip         TEXT
        );
        CREATE TABLE IF NOT EXISTS task_runs (
            task_id            TEXT PRIMARY KEY,
            started            TEXT,
            finished           TEXT,
            status             TEXT,
            total_videos       INTEGER DEFAULT 0,
            processed_videos   INTEGER DEFAULT 0,
            total_frames       INTEGER DEFAULT 0,
            total_detected     INTEGER DEFAULT 0,
            total_snapshots    INTEGER DEFAULT 0,
            total_dedup        INTEGER DEFAULT 0,
            total_no_detection INTEGER DEFAULT 0,
            total_elapsed      REAL    DEFAULT 0,
            details            TEXT
        );
        CREATE TABLE IF NOT EXISTS video_runs (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id          TEXT,
            video            TEXT,
            skipped          INTEGER DEFAULT 0,
            duration         REAL    DEFAULT 0,
            frames_extracted INTEGER DEFAULT 0,
            frames_detected  INTEGER DEFAULT 0,
            snapshots        INTEGER DEFAULT 0,
            dedup_skipped    INTEGER DEFAULT 0,
            no_detection     INTEGER DEFAULT 0,
            elapsed          REAL    DEFAULT 0,
            date             TEXT,
            cancelled        INTEGER DEFAULT 0
        );
        CREATE INDEX IF NOT EXISTS idx_video_runs_task ON video_runs(task_id);
        """
    )
    _conn.commit()


def _count(table):
    return _conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


def _migrate_from_json_once():
    """一次性迁移旧 JSON 数据（仅当对应表为空且 JSON 存在时导入，不删除源文件）。"""
    # 已处理视频：processed.json → processed_videos
    p = Path(config.WORK_DIR) / "processed.json"
    if p.exists() and _count("processed_videos") == 0:
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            data = None
        if isinstance(data, dict) and data:
            _conn.executemany(
                "INSERT OR REPLACE INTO processed_videos(key, processed_at) VALUES(?,?)",
                [(k, t) for k, t in data.items()])
            _conn.commit()
            print(f"[db] migrated {len(data)} processed video(s) from processed.json")

    # 无对象视频：no_object_videos.json → no_object_videos
    p = Path(config.WORK_DIR) / "no_object_videos.json"
    if p.exists() and _count("no_object_videos") == 0:
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            data = None
        if isinstance(data, list) and data:
            rows = [(v.get("path"), v.get("name"), v.get("duration"),
                     v.get("date")) for v in data]
            _conn.executemany(
                "INSERT OR REPLACE INTO no_object_videos(path, name, duration, date) "
                "VALUES(?,?,?,?)", rows)
            _conn.commit()
            print(f"[db] migrated {len(data)} no-object video(s) from "
                  "no_object_videos.json")

    # 事件索引：index.json → events
    p = Path(config.SNAPSHOTS_DIR) / "index.json"
    if p.exists() and _count("events") == 0:
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            data = None
        if isinstance(data, dict) and data:
            rows = [(base,
                     meta.get("source_video"),
                     meta.get("start_time"),
                     meta.get("obj"),
                     meta.get("pose"),
                     meta.get("snapshot"),
                     meta.get("clip"))
                    for base, meta in data.items()]
            _conn.executemany(
                "INSERT OR REPLACE INTO events(base, source_video, start_time, "
                "obj, pose, snapshot, clip) VALUES(?,?,?,?,?,?,?)", rows)
            _conn.commit()
            print(f"[db] migrated {len(data)} event(s) from index.json")


# ---------- processed_videos ----------


def load_processed():
    """已处理视频记录：{key: processed_at}。key = path|size|mtime 保证文件
    被替换后能重新处理。"""
    conn = _get_conn()
    with _lock:
        rows = conn.execute(
            "SELECT key, processed_at FROM processed_videos").fetchall()
    return {k: t for k, t in rows}


def mark_processed(items):
    """批量写入已处理记录：{key: ts}，已存在则覆盖时间戳。"""
    if not items:
        return
    conn = _get_conn()
    with _lock:
        conn.executemany(
            "INSERT OR REPLACE INTO processed_videos(key, processed_at) "
            "VALUES(?,?)", list(items.items()))
        conn.commit()


# ---------- no_object_videos ----------


def load_no_obj():
    """无对象视频列表：[{path, name, duration, date}, ...]"""
    conn = _get_conn()
    with _lock:
        rows = conn.execute(
            "SELECT path, name, duration, date FROM no_object_videos").fetchall()
    return [{"path": r[0], "name": r[1], "duration": r[2], "date": r[3]}
            for r in rows]


def has_no_obj(path):
    conn = _get_conn()
    with _lock:
        row = conn.execute(
            "SELECT 1 FROM no_object_videos WHERE path=?", (path,)).fetchone()
    return row is not None


def add_no_obj(entry):
    """新增/覆盖一条无对象视频记录。"""
    conn = _get_conn()
    with _lock:
        conn.execute(
            "INSERT OR REPLACE INTO no_object_videos(path, name, duration, date) "
            "VALUES(?,?,?,?)",
            (entry.get("path"), entry.get("name"), entry.get("duration"),
             entry.get("date")))
        conn.commit()


def remove_no_obj(path):
    conn = _get_conn()
    with _lock:
        conn.execute("DELETE FROM no_object_videos WHERE path=?", (path,))
        conn.commit()


def remove_no_obj_many(paths):
    """批量删除无对象视频记录（不存在则忽略）。"""
    conn = _get_conn()
    with _lock:
        conn.executemany(
            "DELETE FROM no_object_videos WHERE path=?",
            [(p,) for p in paths])
        conn.commit()


# ---------- events ----------


def upsert_event(base, meta):
    """记录/更新一个事件 JPG 的源视频元数据。base 为快照文件基名。"""
    conn = _get_conn()
    with _lock:
        conn.execute(
            "INSERT OR REPLACE INTO events(base, source_video, start_time, "
            "obj, pose, snapshot, clip) VALUES(?,?,?,?,?,?,?)",
            (base,
             meta.get("source_video"),
             meta.get("start_time"),
             meta.get("obj"),
             meta.get("pose"),
             meta.get("snapshot"),
             meta.get("clip")))
        conn.commit()


def load_events_index():
    """全部事件元数据：{base: {source_video, start_time, obj, pose, ...}}"""
    conn = _get_conn()
    with _lock:
        rows = conn.execute(
            "SELECT base, source_video, start_time, obj, pose, snapshot, clip "
            "FROM events").fetchall()
    out = {}
    for base, sv, st, obj, pose, snap, clip in rows:
        out[base] = {
            "source_video": sv,
            "start_time": float(st) if st is not None else 0.0,
            "obj": obj,
            "pose": pose,
            "snapshot": snap,
            "clip": clip,
        }
    return out


# ---------- task_runs / video_runs（运行指标，供 UI 展示详细进度） ----------

_TASK_RUN_COLS = ("task_id", "started", "finished", "status",
                  "total_videos", "processed_videos", "total_frames",
                  "total_detected", "total_snapshots", "total_dedup",
                  "total_no_detection", "total_elapsed", "details")


def save_task_run(task_id, **fields):
    """插入/覆盖一条任务运行汇总记录（details 存 JSON 文本）。"""
    conn = _get_conn()
    data = {}
    for c in _TASK_RUN_COLS:
        if c in ("started", "finished", "status", "details"):
            data[c] = fields.get(c)
        else:
            data[c] = fields.get(c, 0)
    data["task_id"] = task_id
    with _lock:
        conn.execute(
            "INSERT OR REPLACE INTO task_runs(" + ",".join(_TASK_RUN_COLS) + ") "
            "VALUES(" + ",".join("?" for _ in _TASK_RUN_COLS) + ")",
            [data[c] for c in _TASK_RUN_COLS])
        conn.commit()


def add_video_run(task_id, meta):
    """记录一个视频的处理明细。meta 为 video_done 阶段的 extra dict。"""
    conn = _get_conn()
    with _lock:
        conn.execute(
            "INSERT INTO video_runs(task_id, video, skipped, duration, "
            "frames_extracted, frames_detected, snapshots, dedup_skipped, "
            "no_detection, elapsed, date, cancelled) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
            (task_id,
             meta.get("video"),
             1 if meta.get("skipped") else 0,
             meta.get("duration", 0),
             meta.get("frames_extracted", 0),
             meta.get("frames_detected", 0),
             meta.get("snapshots", 0),
             meta.get("dedup_skipped", 0),
             meta.get("no_detection", 0),
             meta.get("elapsed", 0),
             meta.get("date", ""),
             1 if meta.get("cancelled") else 0))
        conn.commit()


def _row_to_task_run(row, include_details=True):
    d = dict(zip(_TASK_RUN_COLS, row))
    if include_details and d.get("details"):
        try:
            d["details"] = json.loads(d["details"])
        except Exception:
            d["details"] = None
    else:
        # 列表/概览场景不需要逐视频明细（可能很大），直接剔除以减小传输体积
        d.pop("details", None)
    return d


def load_task_run(task_id, include_details=True):
    """单条任务运行汇总，无则返回 None。"""
    conn = _get_conn()
    with _lock:
        row = conn.execute(
            "SELECT " + ",".join(_TASK_RUN_COLS) +
            " FROM task_runs WHERE task_id=?", (task_id,)).fetchone()
    return _row_to_task_run(row, include_details=include_details) if row else None


def load_recent_runs(limit=50, include_details=False):
    """最近 N 条任务运行汇总（按开始时间倒序）。

    include_details=False（默认）时跳过 details 解析与返回，
    仅保留概览所需的聚合指标，显著降低接口传输体积。"""
    conn = _get_conn()
    with _lock:
        rows = conn.execute(
            "SELECT " + ",".join(_TASK_RUN_COLS) +
            " FROM task_runs ORDER BY started DESC LIMIT ?", (limit,)).fetchall()
    return [_row_to_task_run(r, include_details=include_details) for r in rows]


def load_video_runs(task_id):
    """某任务下的全部视频处理明细（按完成顺序）。"""
    conn = _get_conn()
    with _lock:
        rows = conn.execute(
            "SELECT video, skipped, duration, frames_extracted, "
            "frames_detected, snapshots, dedup_skipped, no_detection, "
            "elapsed, date, cancelled FROM video_runs "
            "WHERE task_id=? ORDER BY id", (task_id,)).fetchall()
    return [{
        "video": r[0], "skipped": bool(r[1]), "duration": r[2],
        "frames_extracted": r[3], "frames_detected": r[4],
        "snapshots": r[5], "dedup_skipped": r[6], "no_detection": r[7],
        "elapsed": r[8], "date": r[9], "cancelled": bool(r[10]),
    } for r in rows]
