"""Web UI + daily scheduler for videocut.

Single-container app: serves a browser UI to configure the input/output host
folders (mounted into the container), run the pipeline on demand, view logs
and task history, and preview the produced videos/images. A background
scheduler runs the pipeline daily at the configured time."""
import contextlib
import json
import os
import queue
import threading
import uuid
from datetime import datetime
from pathlib import Path

from flask import (Flask, jsonify, render_template, request,
                    send_from_directory, send_file, abort)
from apscheduler.schedulers.background import BackgroundScheduler

from src import config, main as vc_main, scanner

app = Flask(__name__)

# Merge the NAS AI event dashboard (big-screen viewer) into this same app so
# the whole project runs on a single port. It can also run standalone via
# `python -m ui.web_server` (port 5000).
from .web_server import bp as dashboard_bp
app.register_blueprint(dashboard_bp)

WORK_DIR = Path(os.environ.get("WORK_DIR", "/work"))
CONFIG_PATH = WORK_DIR / "config.json"
TASKS_PATH = WORK_DIR / "tasks.json"
LOGS_DIR = WORK_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_CONFIG = {
    "input_sources": [{"path": "/host", "brand": "auto"}],
    "output_dir": "/host/output",
    "work_dir": "/work",
    "det_conf": config.DET_CONF,
    "det_interval": config.DET_INTERVAL,
    "date_source": config.DATE_SOURCE,
    "camera_brand": config.CAMERA_BRAND,
    "use_pose": config.USE_POSE,
    "similarity_threshold": config.SIMILARITY_THRESHOLD,
    "snapshot_ratio": ":".join(str(x) for x in config.SNAPSHOT_RATIO),
    "snapshot_width": config.SNAPSHOT_WIDTH,
    "snapshot_padding": config.SNAPSHOT_PADDING,
    "schedule_enabled": False,
    "schedule_time": "03:00",
    # Watch the input folder and auto-run when new/changed videos appear.
    "watch_enabled": False,
    "watch_interval": 5,
    # 多线程检测：auto 自动按 CPU 核心数，或用户指定整数。
    "worker_threads": "auto",
    # 临时帧保留天数：超过自动删除（已输出快照不删）。0=不清理。
    "frame_retention_days": 7,
    # 每日自动清理过期帧（与定时检测独立运行）。
    "cleanup_enabled": True,
    "cleanup_time": "04:00",
}

state = {"running": False, "current": None, "cancel": False,
         "progress": {"stage": "idle", "text": "空闲", "current": 0, "total": 0},
         "summary": []}
sched = BackgroundScheduler()

# 任务队列：防止多个检测任务并发或丢失。do_run 从队列取出依次执行。
_task_queue = queue.Queue()
_queue_lock = threading.Lock()
_queue_thread = None

# Tracks which video files were already seen by the watcher so it only fires
# when something is genuinely new/changed.
_watch_state = {"seen": set(), "primed": False}


# ---------- persistence ----------

def load_config():
    if CONFIG_PATH.exists():
        try:
            data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            return {**DEFAULT_CONFIG, **data}
        except Exception:
            pass
    return dict(DEFAULT_CONFIG)


def save_config(cfg):
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2, ensure_ascii=False),
                           encoding="utf-8")


def load_tasks():
    if TASKS_PATH.exists():
        try:
            return json.loads(TASKS_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return []


def save_tasks(tasks):
    TASKS_PATH.write_text(json.dumps(tasks, indent=2, ensure_ascii=False),
                          encoding="utf-8")


cfg = load_config()
tasks = load_tasks()


# ---------- pipeline execution ----------

def list_outputs(output_dir):
    """列出输出目录下的快照图片（本项目不再生成视频）。"""
    out = Path(output_dir) / "snapshots"
    if not out.exists():
        return []
    items = []
    for p in sorted(out.iterdir()):
        if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
            items.append({
                "name": p.name,
                "type": "image",
                "size": p.stat().st_size,
                "mtime": datetime.fromtimestamp(p.stat().st_mtime).isoformat(
                    sep=" "),
                "url": f"/preview/snapshots/{p.name}",
            })
    return items


def do_run(task_id, run_cfg):
    state["running"] = True
    state["current"] = task_id
    state["cancel"] = False
    state["progress"] = {"stage": "scan", "text": "启动中...", "current": 0, "total": 0}
    state["summary"] = []
    log_path = LOGS_DIR / f"{task_id}.log"
    started = datetime.now().isoformat(sep=" ")
    update_task(task_id, status="running", started=started,
                log=log_path.name, params=dict(run_cfg))

    def _progress_cb(info):
        state["progress"] = {"stage": info.get("stage", ""),
                             "text": info.get("text", ""),
                             "current": info.get("current", 0),
                             "total": info.get("total", 0)}
        # video_done 阶段收集每个视频的处理摘要
        if info.get("stage") == "video_done" and info.get("extra"):
            state["summary"].append(info["extra"])

    status = "done"
    try:
        with open(log_path, "w", encoding="utf-8") as lf, \
                contextlib.redirect_stdout(lf), \
                contextlib.redirect_stderr(lf):
            # 行缓冲：每次遇到换行立即 flush 到磁盘，否则 stdout 重定向到
            # 文件时会变成全缓冲，前端读到的日志文件长时间为空。
            try:
                lf.reconfigure(line_buffering=True)
            except Exception:
                pass
            print(f"=== videocut 任务启动 {started} ===", flush=True)
            print(f"输入源: {run_cfg.get('input_sources')}", flush=True)
            print(f"输出目录: {run_cfg.get('output_dir')}", flush=True)
            print(f"线程数: {run_cfg.get('worker_threads', 'auto')}", flush=True)
            vc_main.run_pipeline(
                input_dir=run_cfg.get("input_dir"),
                input_sources=run_cfg.get("input_sources"),
                output_dir=run_cfg["output_dir"],
                work_dir=run_cfg["work_dir"],
                det_conf=float(run_cfg["det_conf"]),
                similarity_threshold=int(run_cfg["similarity_threshold"]),
                det_interval=float(run_cfg["det_interval"]),
                date_source=run_cfg["date_source"],
                camera_brand=run_cfg.get("camera_brand", "auto"),
                use_pose=run_cfg["use_pose"],
                cancel_flag=state,
                progress_cb=_progress_cb,
                worker_threads=run_cfg.get("worker_threads", "auto"),
                frame_retention_days=run_cfg.get("frame_retention_days", 7),
            )
        if state["cancel"]:
            status = "cancelled"
    except Exception as e:
        with open(log_path, "a", encoding="utf-8") as lf:
            lf.write(f"\nERROR: {e}\n")
        state["progress"] = {"stage": "error", "text": f"出错: {e}",
                             "current": 0, "total": 0}
        status = "failed"

    outputs = list_outputs(run_cfg["output_dir"])
    update_task(task_id, status=status,
                finished=datetime.now().isoformat(sep=" "),
                outputs=outputs)
    state["running"] = False
    state["current"] = None
    state["cancel"] = False
    if status == "done":
        state["progress"] = {"stage": "done", "text": "已完成", "current": 0, "total": 0}


def _queue_worker():
    """后台队列消费线程：依次从队列取出任务执行，防止并发与丢失。"""
    while True:
        task_id, run_cfg = _task_queue.get()
        try:
            # 标记为 queued→running 由 do_run 处理
            do_run(task_id, run_cfg)
        except Exception as e:
            print(f"[queue] task {task_id} crashed: {e}")
            update_task(task_id, status="failed",
                        finished=datetime.now().isoformat(sep=" "))
            state["running"] = False
            state["current"] = None
        finally:
            _task_queue.task_done()


def start_run(run_cfg=None):
    """提交一个检测任务到队列。如果当前没有运行的任务会立即开始。
    返回 task_id，或 None 表示队列已满（防止无限堆积）。"""
    run_cfg = run_cfg or cfg
    with _queue_lock:
        qsize = _task_queue.qsize()
        if qsize >= 5:
            return None  # 队列满，拒绝
    task_id = datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    tasks.append({
        "id": task_id, "status": "queued", "started": None,
        "finished": None, "log": None, "params": dict(run_cfg),
        "outputs": [],
    })
    save_tasks(tasks)
    _task_queue.put((task_id, run_cfg))

    # 启动队列消费线程（仅一次）
    global _queue_thread
    if _queue_thread is None or not _queue_thread.is_alive():
        _queue_thread = threading.Thread(target=_queue_worker, daemon=True)
        _queue_thread.start()
    return task_id


def update_task(task_id, **fields):
    for tk in tasks:
        if tk["id"] == task_id:
            tk.update(fields)
            break
    save_tasks(tasks)


def reschedule():
    sched.remove_all_jobs()
    if cfg.get("schedule_enabled"):
        try:
            hh, mm = str(cfg["schedule_time"]).split(":")
            sched.add_job(lambda: start_run(cfg), "cron",
                          hour=int(hh), minute=int(mm), id="daily")
        except Exception as e:
            print("schedule error:", e)
    if cfg.get("watch_enabled"):
        try:
            iv = max(1, int(cfg.get("watch_interval", 5)))
            sched.add_job(auto_check, "interval", minutes=iv, id="watch")
            # Reset priming so the next tick re-baselines the seen set.
            _watch_state["primed"] = False
        except Exception as e:
            print("watch schedule error:", e)
    # 每日自动清理过期帧（独立于检测任务）
    if cfg.get("cleanup_enabled", True):
        try:
            hh, mm = str(cfg.get("cleanup_time", "04:00")).split(":")
            sched.add_job(daily_cleanup, "cron",
                          hour=int(hh), minute=int(mm), id="cleanup")
            print(f"[sched] daily cleanup scheduled at {cfg.get('cleanup_time')}")
        except Exception as e:
            print("cleanup schedule error:", e)


def daily_cleanup():
    """每日自动清理过期帧文件。已输出快照不会被删除。"""
    try:
        work_dir = cfg.get("work_dir", str(WORK_DIR))
        retention = int(cfg.get("frame_retention_days", 7))
        print(f"[cleanup] starting daily cleanup (retention={retention}d, "
              f"work_dir={work_dir})")
        result = vc_main.cleanup_old_frames(
            work_dir=work_dir, retention_days=retention)
        print(f"[cleanup] done: {result}")
    except Exception as e:
        print(f"[cleanup] ERROR: {e}")


def auto_check():
    """Watcher tick: if the set of video files in the input folder changed
    since the last tick, auto-run the pipeline."""
    try:
        # 汇总所有输入源的视频
        all_vids = []
        for src in cfg.get("input_sources", []):
            all_vids.extend(scanner.list_all_videos(
                src["path"], brand=src.get("brand", "auto")))
        vids = all_vids
    except Exception as e:
        print("watch error:", e)
        return
    keys = {v["path"] for v in vids}
    if not _watch_state["primed"]:
        _watch_state["seen"] = keys
        _watch_state["primed"] = True
        return
    if keys != _watch_state["seen"]:
        _watch_state["seen"] = keys
        if not state["running"]:
            print("[watch] input changed, auto-running pipeline")
            start_run(cfg)


# ---------- routes ----------

@app.route("/")
def index():
    # The dashboard is the primary screen; the control/config panel lives at
    # /config and is opened from a button on the dashboard.
    return render_template("dashboard.html")


@app.route("/config")
def config_page():
    return render_template("index.html",
                           cfg=cfg, tasks=tasks,
                           previews=list_outputs(cfg["output_dir"]),
                           running=state["running"])


@app.route("/api/config", methods=["GET"])
def get_config():
    # 加 no-cache 头，确保 iframe 内 /config 每次都拿到最新持久化配置，
    # 不会被浏览器缓存成旧值。
    resp = jsonify(cfg)
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp


@app.route("/api/config", methods=["POST"])
def post_config():
    data = request.get_json(force=True)
    # 把快照比例字符串 "3:2" 转成 config 需要的 tuple
    if "snapshot_ratio" in data and isinstance(data["snapshot_ratio"], str):
        try:
            parts = [int(x) for x in data["snapshot_ratio"].split(":")]
            if len(parts) == 2:
                config.SNAPSHOT_RATIO = tuple(parts)
        except ValueError:
            pass
    if "snapshot_width" in data:
        try:
            config.SNAPSHOT_WIDTH = int(data["snapshot_width"])
        except (ValueError, TypeError):
            pass
    if "snapshot_padding" in data:
        try:
            config.SNAPSHOT_PADDING = float(data["snapshot_padding"])
        except (ValueError, TypeError):
            pass
    cfg.update(data)
    save_config(cfg)
    reschedule()
    return jsonify({"ok": True, "cfg": cfg})


@app.route("/api/run", methods=["POST"])
def api_run():
    tid = start_run(cfg)
    if tid is None:
        return jsonify({"ok": False, "msg": "队列已满（最多5个待处理任务）"}), 409
    return jsonify({"ok": True, "task_id": tid})


@app.route("/api/cancel", methods=["POST"])
def api_cancel():
    if not state["running"]:
        return jsonify({"ok": False, "msg": "not running"}), 409
    state["cancel"] = True
    print("[ui] cancel requested for current task")
    return jsonify({"ok": True})


@app.route("/api/tasks", methods=["GET"])
def api_tasks():
    return jsonify({"tasks": tasks, "running": state["running"],
                    "current": state["current"],
                    "queue_size": _task_queue.qsize()})


@app.route("/api/cleanup", methods=["POST"])
def api_cleanup():
    """手动触发清理过期帧。"""
    try:
        work_dir = cfg.get("work_dir", str(WORK_DIR))
        retention = int(cfg.get("frame_retention_days", 7))
        result = vc_main.cleanup_old_frames(
            work_dir=work_dir, retention_days=retention)
        return jsonify({"ok": True, "result": result})
    except Exception as e:
        return jsonify({"ok": False, "msg": str(e)}), 500


@app.route("/api/progress", methods=["GET"])
def api_progress():
    return jsonify(state["progress"])


@app.route("/api/summary", methods=["GET"])
def api_summary():
    return jsonify({"summary": state["summary"], "running": state["running"]})


@app.route("/api/task/<task_id>", methods=["GET"])
def api_task(task_id):
    tk = next((t for t in tasks if t["id"] == task_id), None)
    log = ""
    if tk and tk.get("log"):
        lp = LOGS_DIR / tk["log"]
        if lp.exists():
            log = lp.read_text(encoding="utf-8", errors="replace")
    return jsonify({"task": tk, "log": log})


@app.route("/api/browse")
def api_browse():
    path = request.args.get("path", "/host")
    p = Path(path)
    if not p.exists() or not p.is_dir():
        return jsonify({"path": path, "dirs": [], "exists": False})
    dirs = sorted([str(d) for d in p.iterdir() if d.is_dir()])
    return jsonify({"path": path, "dirs": dirs, "exists": True})


@app.route("/api/scan")
def api_scan():
    input_dir = request.args.get("input_dir") or (
        cfg["input_sources"][0]["path"] if cfg.get("input_sources") else cfg.get("input_dir", "/host"))
    brand = request.args.get("brand", "auto")
    p = Path(input_dir)
    exists = p.exists() and p.is_dir()
    videos = scanner.list_all_videos(input_dir, brand=brand) if exists else []
    day = scanner.target_date()
    today = [v for v in videos if v["date"] and v["date"][:10] == str(day)]
    return jsonify({
        "input_dir": input_dir,
        "exists": exists,
        "total": len(videos),
        "today": len(today),
        "videos": videos[:200],
    })


@app.route("/api/previews")
def api_previews():
    return jsonify(list_outputs(cfg["output_dir"]))


@app.route("/preview/<path:filename>")
def preview(filename):
    return send_from_directory(cfg["output_dir"], filename)


# ---------- 无对象视频管理 ----------

NO_OBJ_PATH = WORK_DIR / "no_object_videos.json"


def load_no_obj():
    if NO_OBJ_PATH.exists():
        try:
            return json.loads(NO_OBJ_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return []


def save_no_obj(data):
    NO_OBJ_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False),
                           encoding="utf-8")


@app.route("/api/no_object_videos", methods=["GET"])
def api_no_obj_list():
    videos = load_no_obj()
    # 过滤掉已不存在的文件，补充文件大小
    result = []
    for v in videos:
        try:
            p = Path(v["path"])
            if p.exists():
                stat = p.stat()
                result.append({
                    "path": v["path"],
                    "name": v.get("name", p.name),
                    "size_mb": round(stat.st_size / 1024 / 1024, 1),
                    "duration": v.get("duration", "-"),
                    "date": v.get("date", "-"),
                })
        except Exception:
            pass
    return jsonify({"videos": result})


@app.route("/api/no_object_videos", methods=["DELETE"])
def api_no_obj_delete():
    data = request.get_json(force=True, silent=True) or {}
    paths = data.get("paths", [])
    deleted = 0
    errors = []
    for path_str in paths:
        try:
            p = Path(path_str)
            if p.exists():
                p.unlink()
                deleted += 1
        except Exception as e:
            errors.append(str(e))
    # 从记录中移除已删除的
    remaining = [v for v in load_no_obj() if v["path"] not in paths]
    save_no_obj(remaining)
    msg = f"已删除 {deleted} 个视频"
    if errors:
        msg += f"，{len(errors)} 个失败"
    return jsonify({"message": msg, "deleted": deleted, "errors": errors})


@app.route("/api/video")
def api_video():
    """流式播放视频文件。"""
    path_str = request.args.get("path", "")
    if not path_str:
        abort(400)
    try:
        p = Path(path_str).resolve()
        # 安全检查：路径必须在某个输入源目录下
        allowed = False
        for src in cfg.get("input_sources", []):
            try:
                base = Path(src.get("path", "")).resolve()
                p.relative_to(base)
                allowed = True
                break
            except (ValueError, RuntimeError):
                continue
        if not allowed:
            abort(403)
        if not p.exists() or not p.is_file():
            abort(404)
        return send_file(str(p), mimetype="video/mp4")
    except Exception:
        abort(403)


# ---------- bootstrap ----------

reschedule()
if not sched.running:
    sched.start()


import sys
# `python -m ui.app` sets __name__ to "ui.app" (not "__main__"), so detect that
# the module is the entry point via sys.modules identity. `import ui.app`
# elsewhere must NOT start the server.
if __name__ == "__main__" or sys.modules.get("__main__") is sys.modules.get(__name__):
    app.run(host="0.0.0.0", port=8080)
