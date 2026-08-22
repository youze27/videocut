"""NAS AI event dashboard (WebUI).

Serves the big-screen viewer that reads the per-event snapshots/ produced by
the pipeline. Implemented as a Flask Blueprint so it can be merged into the
main control app (single port) and also run standalone on :5000.

从本次需求调整起，事件不再产出裁剪后的 MP4，只保存 JPG 关键帧。每张 JPG
在 SNAPSHOTS_DIR/index.json 中记录其源视频路径与起始时间，大屏点击图片时
通过 /source 路由直接跳转到源视频的对应位置播放（HTML5 video，无需插件）。

文件命名： <YYYYMMDD_HHMMSS>_<obj>_pose-<pose>_<rand>.jpg
  obj  : person | cat | dog | horse | bird | sheep | cow
  pose : lie | stand | sit | squat | bend | unknown | none
"""
import json
import re
from datetime import datetime
from pathlib import Path

from flask import (Blueprint, Flask, jsonify, render_template, request,
                   send_from_directory, send_file, abort)

from src import config

bp = Blueprint("dashboard", __name__, template_folder="templates")

_NAME_RE = re.compile(
    r"(\d{8})_(\d{6})_(person|cat|dog|horse|bird|sheep|cow)_pose-([A-Za-z]+)_(.+)\.jpg$")


def parse_name(name):
    m = _NAME_RE.match(name)
    if not m:
        return None
    d8, t6, obj, pose, rand = m.groups()
    # 兼容旧数据：fall → lie
    if pose == "fall":
        pose = "lie"
    time_full = f"{d8[0:4]}-{d8[4:6]}-{d8[6:8]} {t6[0:2]}:{t6[2:4]}:{t6[4:6]}"
    time_short = f"{d8[4:6]}-{d8[6:8]} {t6[0:2]}:{t6[2:4]}"
    # Unix 时间戳（秒），由服务器端解析，避免浏览器时区偏差
    try:
        dt = datetime.strptime(d8 + t6, "%Y%m%d%H%M%S")
        ts = int(dt.timestamp())
    except Exception:
        ts = 0
    return {
        "d8": d8, "t6": t6, "obj": obj, "pose": pose,
        "rand": rand, "base": f"{d8}_{t6}_{obj}_pose-{pose}_{rand}",
        "timeFull": time_full, "timeShort": time_short,
        "ts": ts,
    }


def _load_events_index():
    """加载 index.json：{base: {source_video, start_time, ...}}"""
    p = Path(config.SNAPSHOTS_DIR) / "index.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def collect_events(scope="all"):
    """收集所有事件 JPG，并附加源视频元数据。

    scope: "all"（全部事件）或 "today"（仅当天事件）。统计范围与事件列表同步。
    """
    index = _load_events_index()
    events = []
    d = Path(config.SNAPSHOTS_DIR)
    if not d.exists():
        return events, {"total": 0, "person": 0, "cat": 0, "dog": 0}

    today_str = datetime.now().strftime("%Y%m%d")
    for p in sorted(d.iterdir()):
        if not p.is_file() or p.suffix.lower() != ".jpg":
            continue
        info = parse_name(p.name)
        if not info:
            continue
        # scope=today 时只保留当天事件
        if scope == "today" and not info["base"].startswith(today_str + "_"):
            continue
        meta = index.get(info["base"], {})
        events.append({
            "timeFull": info["timeFull"],
            "timeShort": info["timeShort"],
            "ts": info["ts"],
            "objType": info["obj"],
            "poseType": info["pose"],
            "sortKey": info["d8"] + info["t6"],
            "jpg": p.name,
            "sourceVideo": meta.get("source_video"),
            "startTime": float(meta.get("start_time", 0) or 0),
        })

    events.sort(key=lambda e: e["sortKey"], reverse=True)

    stats = {"total": 0, "person": 0, "cat": 0, "dog": 0}
    for e in events:
        stats["total"] += 1
        if e["objType"] in stats:
            stats[e["objType"]] += 1
    return events, stats


@bp.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")


@bp.route("/api/events")
def api_events():
    scope = request.args.get("scope", "all")
    if scope not in ("all", "today"):
        scope = "all"
    events, stats = collect_events(scope=scope)
    return jsonify({"events": events, "stats": stats, "scope": scope})


@bp.route("/snap/<path:name>")
def snap(name):
    return send_from_directory(config.SNAPSHOTS_DIR, name)


@bp.route("/clip/<path:name>")
def clip(name):
    """提供裁剪片段 mp4 供大屏播放。从 SNAPSHOTS_DIR/clips/ 读取。"""
    clips_dir = Path(config.SNAPSHOTS_DIR) / "clips"
    # 安全校验：防止路径遍历
    try:
        root = clips_dir.resolve()
        f = (clips_dir / name).resolve()
        f.relative_to(root)
    except (ValueError, RuntimeError):
        abort(403, "forbidden")
    if not f.exists() or not f.is_file():
        abort(404, "clip not found")
    return send_file(str(f), mimetype="video/mp4")


@bp.route("/api/download/<path:name>")
def download_snap(name):
    """下载快照图片。返回 Content-Disposition: attachment 强制浏览器下载。

    支持的查询参数：
      ?download=1  强制下载（默认行为）
      ?view=1      在线预览（inline，与 /snap 等效）
    """
    # 安全校验：解析后路径必须仍在 SNAPSHOTS_DIR 内（防止 ../ 路径遍历）
    try:
        snap_root = Path(config.SNAPSHOTS_DIR).resolve()
        d = (snap_root / name).resolve()
        d.relative_to(snap_root)
    except (ValueError, RuntimeError):
        abort(403, "forbidden: outside snapshots dir")
    if not d.exists() or not d.is_file():
        abort(404, "snapshot not found")
    if request.args.get("view"):
        return send_file(str(d), mimetype="image/jpeg")
    # 默认强制下载
    return send_file(str(d), mimetype="image/jpeg",
                     as_attachment=True, download_name=d.name)


@bp.route("/api/download")
def download_list():
    """列出所有可下载的快照图片，供外部调用获取列表后逐个下载。"""
    d = Path(config.SNAPSHOTS_DIR)
    if not d.exists():
        return jsonify({"items": [], "total": 0})
    items = []
    for p in sorted(d.iterdir()):
        if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png"):
            items.append({
                "name": p.name,
                "url": f"/api/download/{p.name}",
                "preview": f"/snap/{p.name}",
                "size": p.stat().st_size,
            })
    return jsonify({"items": items, "total": len(items)})


@bp.route("/source")
def source():
    """提供源视频文件用于大屏跳转播放。仅允许访问 INPUT_DIR 下的文件。"""
    raw = request.args.get("path", "")
    if not raw:
        abort(400, "missing path")
    p = Path(raw)
    try:
        p_resolved = p.resolve()
        input_root = Path(config.INPUT_DIR).resolve()
    except Exception:
        abort(400, "invalid path")
    # 路径安全校验：只允许访问 INPUT_DIR 下的文件（防止路径遍历）
    try:
        p_resolved.relative_to(input_root)
    except ValueError:
        abort(403, "forbidden: outside INPUT_DIR")
    if not p_resolved.exists() or not p_resolved.is_file():
        abort(404, "not found")
    return send_from_directory(p_resolved.parent, p_resolved.name,
                               conditional=True)


def create_app():
    app = Flask(__name__, template_folder="templates")
    app.register_blueprint(bp)
    return app


if __name__ == "__main__":
    create_app().run(host="0.0.0.0", port=5000)
