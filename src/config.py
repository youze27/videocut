"""Runtime configuration with sensible defaults. Values can be overridden
via environment variables (see entrypoint / docker-compose).

注意：本项目不再生成合并视频或拼图，只扫描视频、抽帧检测人/宠、为每个
检测到的事件保存一张裁剪到 3:2 的 JPG 快照（带姿态标签）。"""
import os
from pathlib import Path

INPUT_DIR = Path(os.environ.get("INPUT_DIR", "/input"))
# 支持多个输入源（不同摄像头文件夹），每个源可独立指定摄像头品牌用于时间戳解析。
# 格式: [{"path": "/host/cam1", "brand": "xiaomi"}, {"path": "/host/cam2", "brand": "auto"}]
# 运行时由 UI 传入 JSON；环境变量 INPUT_DIR 仍保留作为单目录回退。
INPUT_SOURCES = []
_env_input = os.environ.get("INPUT_DIR", "")
if _env_input:
    # 环境变量支持逗号分隔多目录，brand 默认 auto
    for d in _env_input.split(","):
        d = d.strip()
        if d:
            INPUT_SOURCES.append({"path": d, "brand": "auto"})
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "/output"))
WORK_DIR = Path(os.environ.get("WORK_DIR", "/work"))

# Where YOLOv8 weights are downloaded at build time.
MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/app/models"))
# 本地开发回退：容器路径不存在时，用仓库内 models/ 目录。
# 注意：Windows 上 /app/models 可能命中残留的 C:\app\models（旧容器产物），
# 导致加载到过期的 .pt 而非 OpenVINO IR。因此只要仓库内 models/ 含
# 当前 IR（yolov8n-pose_int8_openvino_model），一律优先使用仓库目录。
# 容器内 /app/models 与仓库 models/ 是同一份拷贝（Dockerfile COPY），结果一致。
_local_models = Path(__file__).resolve().parent.parent / "models"
if _local_models.exists() and (
        _local_models / "yolov8n-pose_int8_openvino_model").exists():
    MODEL_DIR = _local_models
elif not MODEL_DIR.exists():
    if _local_models.exists():
        MODEL_DIR = _local_models

# Length (seconds) of each candidate clip cut from source videos (用于抽帧检测).
SEGMENT_DURATION = int(os.environ.get("SEGMENT_DURATION", "4"))

# 扫描模式：
#   "smart"  — 智能扫描（默认）：先按扩展名+文件夹名/文件名快速过滤当天视频，
#              再只取最近 SCAN_PROCESS_HOURS 小时的视频处理。扫描量最小、速度最快。
#   "latest" — 每个输入目录取最近 SCAN_LATEST_COUNT 个未处理视频
#   "hours"  — 处理最近 SCAN_HOURS 小时的视频
#   "date"   — 按日期过滤（DATE_TOLERANCE_DAYS）
SCAN_MODE = os.environ.get("SCAN_MODE", "smart")

# 只处理最近 N 小时内录制的视频（SCAN_MODE=hours 时生效）。
# 设为 0 表示禁用此过滤、回退到按 DATE_TOLERANCE_DAYS 的日期过滤。
SCAN_HOURS = float(os.environ.get("SCAN_HOURS", "2"))

# SCAN_MODE=latest 时，每个输入目录最多取最近 N 个未处理视频。
SCAN_LATEST_COUNT = int(os.environ.get("SCAN_LATEST_COUNT", "30"))

# ===== smart 扫描模式参数 =====
# 扫描范围：只扫描当天的视频（按文件夹名或文件名时间戳判断），减少扫描量。
SCAN_TODAY_ONLY = os.environ.get("SCAN_TODAY_ONLY", "true").lower() == "true"
# 处理范围：只处理最近 N 小时的视频（从扫描结果中再过滤），默认 2 小时。
# 设为 0 表示处理当天所有扫描到的视频。
SCAN_PROCESS_HOURS = float(os.environ.get("SCAN_PROCESS_HOURS", "2"))

# How many days back from today count as "today" (0 = strictly today).
DATE_TOLERANCE_DAYS = int(os.environ.get("DATE_TOLERANCE_DAYS", "0"))

# Person detection confidence threshold.
DET_CONF = float(os.environ.get("DET_CONF", "0.5"))

# Sampling interval (seconds) between frames extracted for detection.
DET_INTERVAL = float(os.environ.get("DET_INTERVAL", "3.0"))

# 唯一模型：yolov8n-pose.pt（COCO-pose）。一次推理同时输出 person 检测框
# (boxes) 与 17 骨骼关键点 (keypoints)。全局固定使用，无双模型切换。
# 部署形态：默认加载 INT8 OpenVINO IR（体积小 40%、批推理再快约 5~8%，
# 精度几乎无损，已验证 IoU≈0.97）；FP16 IR 保留为备选。
# 支持路径：OpenVINO 目录/onnx 文件/.pt 权重。
MODEL_PATH = Path(os.environ.get("MODEL_PATH", str(MODEL_DIR / "yolov8n-pose_int8_openvino_model")))
# 兼容旧环境变量 POSE_MODEL：未设置 MODEL_PATH 时回退。
if not os.environ.get("MODEL_PATH"):
    MODEL_PATH = Path(os.environ.get("POSE_MODEL", str(MODEL_PATH)))

# Frames sampled per segment for person detection (legacy, kept for compat).
DET_SAMPLES = int(os.environ.get("DET_SAMPLES", "8"))

# How to decide a video's date: "mtime", "filename", or "either"
# (filename parses camera timestamps; see timestamp_parser).
DATE_SOURCE = os.environ.get("DATE_SOURCE", "either")
# Camera brand for filename timestamp parsing: auto/xiaomi/hikvision/dahua/generic.
CAMERA_BRAND = os.environ.get("CAMERA_BRAND", "auto")

# Hamming-distance threshold for perceptual-hash visual dedup.
# Smaller = stricter (more things considered duplicates).
# 默认 5：监控场景背景固定，pHash 偏向低频结构，阈值过大易误判不同对象为重复。
SIMILARITY_THRESHOLD = int(os.environ.get("SIMILARITY_THRESHOLD", "5"))

# Per-event outputs (NAS event dashboard). Snapshots = AI key-frame JPGs.
# Naming: <YYYYMMDD_HHMMSS>_<obj>_pose-<pose>_<rand>.jpg
SNAPSHOTS_DIR = Path(os.environ.get("SNAPSHOTS_DIR", str(OUTPUT_DIR / "snapshots")))

# 始终输出事件快照（本项目已不再生成合并视频）。
EMIT_EVENTS = os.environ.get("EMIT_EVENTS", "true").lower() == "true"

# 事件快照裁剪：围绕检测框裁剪突出主体，统一比例与尺寸，大屏展示更整齐。
# SNAPSHOT_RATIO 形如 "3:2"（宽:高）；SNAPSHOT_WIDTH 为输出宽度（px），高度按比例计算。
SNAPSHOT_RATIO = tuple(int(x) for x in os.environ.get("SNAPSHOT_RATIO", "3:2").split(":"))
SNAPSHOT_WIDTH = int(os.environ.get("SNAPSHOT_WIDTH", "800"))
# 裁剪区域相对检测框的扩展比例（0.25 = 检测框较大边的 25% 作为额外边距）。
SNAPSHOT_PADDING = float(os.environ.get("SNAPSHOT_PADDING", "0.1"))

# 帧文件保留天数：超过此天数的临时帧（WORK_DIR/frames 下）自动删除。
# 已输出的快照（SNAPSHOTS_DIR 下）不会删除。0 = 不自动清理。
FRAME_RETENTION_DAYS = int(os.environ.get("FRAME_RETENTION_DAYS", "30"))

# 多线程检测：处理多个视频时并行的线程数。
# "auto" = 自动按 CPU 核心数（最多 4），或用户指定整数。0/1 = 单线程。
WORKER_THREADS = os.environ.get("WORKER_THREADS", "auto")

# 检测完成后自动生成网格截图（后端 PIL 拼接当天全部快照）。
# 保存到 OUTPUT_DIR/grid_latest.jpg，前端通过 /api/grid_latest 访问。
# 固定 5 列、最多 100 张、按时间正序排列；GRID_COLS 控制列数。
GRID_COLS = int(os.environ.get("GRID_COLS", "5"))
GRID_ROWS = int(os.environ.get("GRID_ROWS", "20"))

# 网格截图压缩（另存一份 KB 级压缩图，供轻量传输/外部接口使用）：
# - GRID_COMPRESS: 是否在生成 grid_latest.jpg 时额外生成 grid_latest_compressed.jpg
# - GRID_MAX_WIDTH: 压缩图最大宽度（px），按此等比缩放，控制整体尺寸
# - GRID_TARGET_KB: 压缩目标体积（KB），JPEG 质量会自适配到该上限以下
# - GRID_COMPRESS_MIN_Q: 质量下限，防止为了压体积把图压到看不清
GRID_COMPRESS = bool(int(os.environ.get("GRID_COMPRESS", "1")))
GRID_MAX_WIDTH = int(os.environ.get("GRID_MAX_WIDTH", "1280"))
GRID_TARGET_KB = int(os.environ.get("GRID_TARGET_KB", "200"))
GRID_COMPRESS_MIN_Q = int(os.environ.get("GRID_COMPRESS_MIN_Q", "40"))

# 裁剪片段：检测到事件时，从源视频裁剪前后 CLIP_SECONDS 秒的短视频。
# 保存到 SNAPSHOTS_DIR/clips/，与快照同名（.mp4）。点击快照时播放对应片段。
# 凌晨定时将前一天的片段按时间顺序拼接为 YYYYMMDD.mp4。
CLIP_SECONDS = float(os.environ.get("CLIP_SECONDS", "5"))
