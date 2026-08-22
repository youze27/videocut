"""Web UI + daily scheduler for videocut.

Single-container app: serves a browser UI to configure the input/output host
folders (mounted into the container), run the pipeline on demand, view logs
and task history, and preview the produced videos/images. A background
scheduler runs the pipeline daily at the configured time."""
import contextlib
import json
import os
import queue
import sys
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

from flask import (Flask, jsonify, render_template, request,
                    send_from_directory, send_file, abort, make_response)
from apscheduler.schedulers.background import BackgroundScheduler

from src import config, main as vc_main, scanner

app = Flask(__name__)
# 强制禁用模板缓存，确保每次请求都从磁盘读取最新模板
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.jinja_env.auto_reload = True

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
    # 扫描模式：smart（只扫当天+处理最近N小时）/ latest（最近N个）/ hours / date
    "scan_mode": "smart",
    "scan_today_only": True,
    "scan_process_hours": 2,
    "scan_latest_count": 30,
    "snapshot_ratio": ":".join(str(x) for x in config.SNAPSHOT_RATIO),
    "snapshot_width": config.SNAPSHOT_WIDTH,
    "snapshot_padding": config.SNAPSHOT_PADDING,
    # 定时检测：按间隔小时数自动运行，默认启用，每 1 小时运行一次
    "schedule_enabled": True,
    "schedule_interval_hours": 1,
    # Watch the input folder and auto-run when new/changed videos appear.
    # 默认启用：服务/容器启动即进入自动运行状态，无需人为操作。
    "watch_enabled": True,
    # watch 自动运行间隔（分钟），默认 1 小时
    "watch_interval": 60,
    # 多线程检测：auto 自动按 CPU 核心数，或用户指定整数。
    "worker_threads": "auto",
    # 临时帧保留天数：超过自动删除（已输出快照不删）。0=不清理。默认 30 天
    "frame_retention_days": 30,
    # 每日自动清理过期帧（与定时检测独立运行）。
    "cleanup_enabled": True,
    "cleanup_time": "04:00",
    # 网格截图：后端 PIL 拼接当天全部快照（5 列、最多 100 张、按时间正序）。
    "grid_cols": config.GRID_COLS,
    "grid_rows": config.GRID_ROWS,
    # 任务最长运行时间（小时），超过自动杀死。0=不限制。默认 24 小时
    "max_task_hours": 24,
}

# 任务状态：generation 用于让被杀死的旧 worker 失效，防止它结束后又改 state。
# force_kill: 硬杀死标志；task_start: 当前任务开始时间戳。
state = {"running": False, "current": None, "cancel": False,
         "force_kill": False, "task_start": None, "generation": 0,
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


def _apply_config_to_module(data):
    """把配置字典同步到 config 模块（pipeline 运行时从模块读取）。

    启动时和 API 保存配置时都要调用，确保持久化配置生效。
    """
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
    for k in ("grid_cols", "grid_rows"):
        if k in data:
            try:
                setattr(config, k.upper(), max(1, int(data[k])))
            except (ValueError, TypeError):
                pass
    if "scan_mode" in data:
        config.SCAN_MODE = str(data["scan_mode"])
    if "scan_today_only" in data:
        config.SCAN_TODAY_ONLY = bool(data["scan_today_only"])
    if "scan_process_hours" in data:
        try:
            config.SCAN_PROCESS_HOURS = float(data["scan_process_hours"])
        except (ValueError, TypeError):
            pass
    if "scan_latest_count" in data:
        try:
            config.SCAN_LATEST_COUNT = int(data["scan_latest_count"])
        except (ValueError, TypeError):
            pass


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
_apply_config_to_module(cfg)  # 启动时把持久化配置同步到 config 模块
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
    my_generation = state["generation"]  # 记录代次，被杀死后此 worker 的状态更新失效
    state["running"] = True
    state["current"] = task_id
    state["cancel"] = False
    state["force_kill"] = False
    state["task_start"] = time.time()
    state["progress"] = {"stage": "scan", "text": "启动中...", "current": 0, "total": 0}
    state["summary"] = []
    log_path = LOGS_DIR / f"{task_id}.log"
    started = datetime.now().isoformat(sep=" ")
    update_task(task_id, status="running", started=started,
                log=log_path.name, params=dict(run_cfg))

    def _is_active():
        """当前 worker 是否仍掌握 state 控制权（未被杀死/超时取代）。"""
        return (state["generation"] == my_generation
                and state["current"] == task_id)

    def _progress_cb(info):
        # 被杀死后不再接收进度，避免覆盖新任务的状态
        if not _is_active():
            return
        state["progress"] = {"stage": info.get("stage", ""),
                             "text": info.get("text", ""),
                             "current": info.get("current", 0),
                             "total": info.get("total", 0),
                             "extra": info.get("extra")}
        # video_done 阶段收集每个视频的处理摘要
        if info.get("stage") == "video_done" and info.get("extra"):
            state["summary"].append(info["extra"])

    status = "done"
    try:
        with open(log_path, "w", encoding="utf-8") as lf, \
                contextlib.redirect_stdout(lf), \
                contextlib.redirect_stderr(lf):
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
                scan_mode=run_cfg.get("scan_mode", "smart"),
            )
        if state["cancel"]:
            status = "cancelled"
    except Exception as e:
        with open(log_path, "a", encoding="utf-8") as lf:
            lf.write(f"\nERROR: {e}\n")
        if _is_active():
            state["progress"] = {"stage": "error", "text": f"出错: {e}",
                                 "current": 0, "total": 0}
        status = "failed"

    outputs = list_outputs(run_cfg["output_dir"]) if _is_active() else []
    # 仅当此 worker 仍活跃时才更新任务记录与全局状态；
    # 已被杀死/超时（generation 变化）的 worker 不再改写 state，避免 resurrect。
    if _is_active():
        update_task(task_id, status=status,
                    finished=datetime.now().isoformat(sep=" "),
                    outputs=outputs)
        state["running"] = False
        state["current"] = None
        state["cancel"] = False
        state["force_kill"] = False
        state["task_start"] = None
        if status == "done":
            state["progress"] = {"stage": "done", "text": "已完成", "current": 0, "total": 0}
    else:
        # 被取代：只补 outputs（若仍活跃），不触碰 state
        update_task(task_id, outputs=outputs)


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


def _force_kill_task(reason="killed"):
    """硬终止当前运行的任务。

    与软取消（cancel=True，等 pipeline 在检查点自行退出）不同，force kill 会：
    1. 推进 generation，让旧 worker 结束时不再改写 state（避免 resurrect）
    2. 立即重置 state，允许下一个任务启动
    3. 把任务记录标记为 killed/timeout

    注意：真正卡在子进程（如 ffmpeg hang）的 worker 线程无法被 Python 层面杀死，
    会继续在后台跑到自己退出，但已不占用 state、不阻塞新任务。
    """
    tid = state.get("current")
    state["generation"] += 1
    state["running"] = False
    state["current"] = None
    state["cancel"] = True          # 同时置 cancel，让 pipeline 检查点尽快退出
    state["force_kill"] = True
    state["task_start"] = None
    state["progress"] = {"stage": "idle", "text": f"已终止（{reason}）",
                         "current": 0, "total": 0}
    state["summary"] = []
    if tid:
        update_task(tid, status=reason,
                    finished=datetime.now().isoformat(sep=" "))
        print(f"[kill] task {tid} force-killed (reason={reason})")
    # 清空队列里未开始的排队任务，避免杀掉一个后又立刻起一个
    cleared = 0
    while True:
        try:
            t_id, _ = _task_queue.get_nowait()
            update_task(t_id, status="cancelled",
                        finished=datetime.now().isoformat(sep=" "))
            _task_queue.task_done()
            cleared += 1
        except queue.Empty:
            break
    if cleared:
        print(f"[kill] cleared {cleared} queued task(s)")


def _watchdog():
    """后台监控线程：每分钟检查当前任务运行时长，超过 max_task_hours 自动杀死。

    服务/容器重启后也会把 tasks.json 中残留的 running/queued 任务标记为 failed，
    避免它们永久卡在运行状态（这些任务实际上已经随重启死掉了）。
    """
    while True:
        time.sleep(60)
        try:
            # 超时检查
            if state["running"] and state.get("task_start"):
                max_hours = float(cfg.get("max_task_hours", 24) or 0)
                if max_hours > 0:
                    elapsed = time.time() - state["task_start"]
                    if elapsed > max_hours * 3600:
                        print(f"[watchdog] task {state['current']} ran for "
                              f"{elapsed/3600:.1f}h > {max_hours}h, killing")
                        _force_kill_task("timeout")
        except Exception as e:
            print(f"[watchdog] error: {e}")


def _cleanup_zombie_tasks():
    """服务启动时把 tasks.json 中残留的 running/queued 任务标记为 failed。

    容器重启意味着这些任务已死，但记录里仍是 running 状态，导致前端一直显示运行中。
    """
    now = datetime.now().isoformat(sep=" ")
    changed = False
    for t in tasks:
        if t.get("status") in ("running", "queued"):
            t["status"] = "failed"
            t["finished"] = t.get("finished") or now
            t["error"] = "服务重启，任务中断"
            changed = True
            print(f"[startup] marked zombie task {t['id']} as failed")
    if changed:
        save_tasks(tasks)


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
    # 任务历史最多保留 100 条（超出则丢弃最旧的）
    if len(tasks) > 100:
        del tasks[:-100]
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
    # 定时检测：按间隔小时数自动运行（如每 2 小时运行一次）
    if cfg.get("schedule_enabled"):
        try:
            hours = float(cfg.get("schedule_interval_hours", 2))
            if hours <= 0:
                hours = 2
            sched.add_job(lambda: start_run(cfg), "interval",
                          hours=hours, id="scheduled")
            print(f"[sched] 定时检测已启用：每 {hours} 小时运行一次")
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
            ct = str(cfg.get("cleanup_time", "04:00") or "04:00")
            parts = ct.split(":")
            if len(parts) != 2:
                ct = "04:00"
                parts = ["04", "00"]
            hh, mm = parts
            sched.add_job(daily_cleanup, "cron",
                          hour=int(hh), minute=int(mm), id="cleanup")
            print(f"[sched] daily cleanup scheduled at {ct}")
        except Exception as e:
            print("cleanup schedule error:", e)
    # 每日凌晨 00:30 拼接前一天的裁剪片段为长视频
    try:
        sched.add_job(daily_concat, "cron",
                      hour=0, minute=30, id="daily_concat")
        print("[sched] daily clip concat scheduled at 00:30")
    except Exception as e:
        print("concat schedule error:", e)


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


def daily_concat():
    """每日凌晨将前一天的裁剪片段按时间顺序拼接为长视频。"""
    try:
        print("[concat] starting daily clip concatenation")
        vc_main.concat_daily_clips()
        print("[concat] done")
    except Exception as e:
        print(f"[concat] ERROR: {e}")


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
        if not state["running"]:
            _watch_state["seen"] = keys
            print("[watch] input changed, auto-running pipeline")
            start_run(cfg)
        # 运行中：保持 seen 不变，留待下个 tick 检测，避免吞掉运行期间落盘的新文件


# ---------- routes ----------

@app.route("/")
def index():
    # The dashboard is the primary screen; the control/config panel lives at
    # /config and is opened from a button on the dashboard.
    resp = make_response(render_template("dashboard.html"))
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp


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
    _apply_config_to_module(data)
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


@app.route("/api/kill", methods=["POST"])
def api_kill():
    """硬杀死当前任务：立即重置 state 并标记任务为 killed，同时清空队列。

    与 /api/cancel（软取消，等 pipeline 检查点退出）不同，kill 用于任务卡死、
    长时间无响应的场景。被杀死的 worker 线程可能仍在后台跑，但不再持有 state。
    """
    if not state["running"]:
        # 即便没在运行，也清一下可能残留的队列
        _force_kill_task("killed")
        return jsonify({"ok": True, "msg": "no running task, queue cleared"})
    _force_kill_task("killed")
    return jsonify({"ok": True, "msg": "task killed"})


@app.route("/api/tasks", methods=["GET"])
def api_tasks():
    # 返回运行状态、当前任务、队列大小、已运行时长（秒）
    elapsed = None
    if state["running"] and state.get("task_start"):
        elapsed = round(time.time() - state["task_start"], 0)
    return jsonify({"tasks": tasks, "running": state["running"],
                    "current": state["current"],
                    "queue_size": _task_queue.qsize(),
                    "elapsed": elapsed})


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
    path = request.args.get("path", "")
    if not path or path == "/host":
        # /host 是 Docker 路径，本地运行时回退到系统根目录
        if sys.platform == "win32":
            # Windows: 列出所有盘符
            drives = [f"{c}:\\" for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                      if Path(f"{c}:\\").exists()]
            return jsonify({"path": "此电脑", "dirs": drives, "exists": True,
                            "is_root": True})
        path = "/"
    p = Path(path)
    if not p.exists() or not p.is_dir():
        return jsonify({"path": path, "dirs": [], "exists": False})
    try:
        dirs = sorted([str(d) for d in p.iterdir() if d.is_dir()
                       and not d.name.startswith(".")])
    except (PermissionError, OSError):
        return jsonify({"path": path, "dirs": [], "exists": True,
                        "error": "无权限访问"})
    return jsonify({"path": str(p), "dirs": dirs, "exists": True})


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


# ---------- 网格截图（检测完成后自动拼接的最新快照网格） ----------

@app.route("/api/grid_latest")
def api_grid_latest():
    """返回最新的网格截图。检测完成后自动生成，固定 URL 供外部访问。"""
    grid_path = Path(cfg["output_dir"]) / "grid_latest.jpg"
    if not grid_path.exists():
        abort(404)
    resp = send_file(str(grid_path), mimetype="image/jpeg")
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return resp


@app.route("/api/grid_upload", methods=["POST"])
def api_grid_upload():
    """保留接口：接收上传的截图并保存为 grid_latest.jpg。

    当前网格截图统一由后端 PIL 拼接当天快照生成（见 main._generate_grid_snapshot），
    前端不再上传。此接口保留以兼容外部调用。"""
    if "file" not in request.files:
        return jsonify({"ok": False, "error": "no file"}), 400
    f = request.files["file"]
    if not f.filename:
        return jsonify({"ok": False, "error": "empty filename"}), 400
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "grid_latest.jpg"
    f.save(str(out_path))
    st = out_path.stat()
    print(f"[grid] uploaded screenshot to {out_path} ({st.st_size/1024:.1f}KB)")
    return jsonify({
        "ok": True,
        "url": "/api/grid_latest",
        "size_kb": round(st.st_size / 1024, 1),
        "updated": datetime.fromtimestamp(st.st_mtime).strftime(
            "%Y-%m-%d %H:%M:%S"),
    })


@app.route("/api/grid_info")
def api_grid_info():
    """返回网格截图的配置和状态信息。"""
    grid_path = Path(cfg["output_dir"]) / "grid_latest.jpg"
    exists = grid_path.exists()
    info = {
        "exists": exists,
        "url": "/api/grid_latest",
        "cols": config.GRID_COLS,
        "rows": config.GRID_ROWS,
        "total_cells": config.GRID_COLS * config.GRID_ROWS,
    }
    if exists:
        st = grid_path.stat()
        info["size_kb"] = round(st.st_size / 1024, 1)
        info["updated"] = datetime.fromtimestamp(st.st_mtime).strftime(
            "%Y-%m-%d %H:%M:%S")
    return jsonify(info)


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

# 启动时清理僵尸任务（容器重启后残留的 running/queued 记录）
_cleanup_zombie_tasks()

reschedule()
if not sched.running:
    sched.start()

# 启动 watchdog 线程：监控任务超时，超过 max_task_hours 自动杀死
threading.Thread(target=_watchdog, daemon=True).start()


def _startup_autorun():
    """服务/容器启动后自动触发首次扫描，实现"启动即运行、全自动化"。
    延迟 3 秒让 HTTP 服务先就绪；watch 模式负责后续的持续自动化。"""
    try:
        time.sleep(3)
        tid = start_run(cfg)
        if tid:
            print(f"[startup] 自动触发首次扫描，task_id={tid}")
        else:
            print("[startup] 队列已满，跳过首次自动扫描")
    except Exception as e:
        print(f"[startup] 自动首次扫描失败: {e}")


threading.Thread(target=_startup_autorun, daemon=True).start()


import sys
# `python -m ui.app` sets __name__ to "ui.app" (not "__main__"), so detect that
# the module is the entry point via sys.modules identity. `import ui.app`
# elsewhere must NOT start the server.
if __name__ == "__main__" or sys.modules.get("__main__") is sys.modules.get(__name__):
    app.run(host="0.0.0.0", port=8080)
