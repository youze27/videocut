# VideoCut 开发文档

面向二次开发者的内部文档。**使用说明、参数表、API 参考请看 [README](../README.md)**，本文只讲架构、代码结构、扩展方式与调试手法。

---

## 📐 1. 整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│  ui/app.py  (Flask, 单进程)                                      │
│  ├─ 页面  /            → 大屏 dashboard.html                    │
│  │        /config      → 配置 config.html                       │
│  │        /tasks       → 任务与日志 tasks.html                   │
│  ├─ API   /api/*       → 配置 / 运行 / 进度 / 图片 / 任务        │
│  ├─ 调度  APScheduler  → 定时运行、帧清理、每日片段拼接          │
│  ├─ 队列  Queue(≤5)    → 串行执行管线，避免 NAS 被 IO 打满        │
│  └─ 看门狗 (60s)       → 超时任务终止 / 僵尸任务清理              │
└──────────────────────────┬──────────────────────────────────────┘
                           │  config.json  (UI 写入 WORK_DIR)
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  src/main.py  run_pipeline() / process()                         │
│  ① scanner   ② dedup(L2/L3)  ③ extract  ④ detect  ⑤ 裁剪      │
│  ⑥ 姿态重试  ⑦ db.upsert_event  ⑧ _generate_grid_snapshot        │
└──┬──────────┬──────────┬──────────┬──────────┬──────────────────┘
   │          │          │          │          │
scanner.py  extract.py  detect.py   dedup.py    db.py
时间解析    ffmpeg 抽帧  YOLO+OpenVINO  pHash    SQLite(WAL)
   │                     │
timestamp_parser.py   utils.py
品牌解析器            原子写入 / pHash / 压缩
```

### 数据流向

| 阶段 | 读 | 写 |
|---|---|---|
| 扫描 | `INPUT_SOURCES[]` 目录树 | — |
| 视频去重 | `processed_videos` 表 | `processed_videos` |
| 抽帧 | 源视频 | `WORK_DIR/frames/<video_uid>/` |
| 推理 | 帧 JPG | — |
| 事件去重 | 进程内 `DedupFilter` | 内存 + `events` 表 |
| 快照产出 | 帧 JPG + bbox + 骨架 | `OUTPUT_DIR/snapshots/*.jpg`、`clips/*.mp4` |
| 网格拼接 | 全部快照 | `OUTPUT_DIR/grid_latest.jpg`、`grid_history/` |

---

## 🗂️ 2. 目录与文件职责

```
videocut/
├── src/                        # 纯后端，无 Flask 依赖，可单独 import 复用
│   ├── config.py               # 所有环境变量集中定义（唯一配置源）
│   ├── main.py                 # 管线编排：process() / _process_one_video() / 网格
│   ├── scanner.py              # 目录遍历 + 快速日期预筛
│   ├── timestamp_parser.py     # 各品牌文件名时间戳解析（注册表模式）
│   ├── extract.py              # ffmpeg 关键帧 / 定距抽帧（含看门狗）
│   ├── detect.py               # 模型加载、批推理、姿态分类、骨架绘制
│   ├── dedup.py                # pHash 帧级去重（DedupFilter）
│   ├── db.py                   # SQLite 封装（六表）
│   └── utils.py                # 原子写入、pHash、JPEG 自适应压缩、ffprobe
├── ui/
│   ├── app.py                  # Flask 应用 + 调度 + 队列 + 看门狗
│   ├── web_server.py           # 独立 WSGI 入口（无 Flask-Run）
│   ├── requirements-ui.txt
│   └── templates/              # 3 个原生 HTML（无构建步骤、无框架）
├── tools/                      # 模型转换与验证（离线执行，非运行时依赖）
│   ├── collect_calib.py        # 采集 INT8 校准集
│   ├── export_model.py         # PT → ONNX → OpenVINO IR（INT8/FP16）
│   └── verify_int8.py          # FP16 vs INT8 精度/延迟对比
├── models/                     # 推理权重（入库），含 calib_data.yaml
├── docs/images/                # README 配图（入库白名单）
├── Dockerfile
├── docker-compose.yml          # NAS 部署
├── docker-compose.local.yml    # Windows/macOS 本地开发
└── dev_server.py               # 本地免容器启动
```

> **原则**：`src/` 不 import `ui/`。任何后端能力都应能在 `python -m src.main` 或直接 import 下跑通，方便写脚本与单测。

---

## 🔌 3. 核心扩展点

### 3.1 新增摄像头品牌

`src/timestamp_parser.py` 采用注册表模式：

```python
from datetime import datetime

class MyBrandParser(BaseTimestampParser):
    brand = "mybrand"
    description = "MyBrand: CAM01_YYYYMMDD_HHMMSS.mp4"

    def parse(self, file_path: Path):
        # 返回 datetime；无法解析返回 None
        m = re.search(r"(\d{8})_(\d{6})", file_path.stem)
        if not m:
            return None
        return datetime.strptime("".join(m.groups()), "%Y%m%d%H%M%S")

register_parser("mybrand", MyBrandParser, aliases=["mb", "my"])
```

要点：
- `aliases` 用于 UI 下拉框的等价写法
- 解析器**只做解析**，不做 IO 统计；扫描性能敏感
- 注册后 UI 的「品牌」下拉框会自动多出一项（`brand="auto"` 时按顺序尝试所有解析器）

### 3.2 快速预筛 vs 权威解析（重要）

两级解析器**顺序相反**，是有意设计：

| | `scanner._quick_date()` | `timestamp_parser.parse_timestamp()` |
|---|---|---|
| 位置 | 扫描早期，逐文件 | 命中后、抽帧前 |
| 手段 | 纯正则匹配路径字符串 | 读文件 + 必要时 ffprobe |
| 目的 | **快速排除**（大批量） | **权威判定**（少量） |
| 失败 | 保守放行 | 记录并跳过 |

> 新增品牌时**只改 `timestamp_parser.py`**。`_quick_date()` 里针对小米命名的特判可以不动 —— 它只是预筛，宽松一点只影响速度，不影响正确性。

### 3.3 新增检测对象

`detect.py` 使用 YOLOv8-pose 单模型，一次推理同时得到 person 检测框与 17 关键点。要新增类别：

1. `classify_pose()` 增加返回分支
2. `main._GRID_POSE_LABEL` / `_GRID_POSE_COLOR` 增加配色
3. `ui/templates/dashboard.html` 的姿态筛选下拉框增加选项

### 3.4 调整网格拼接规则

全部逻辑集中在 `main._generate_grid_snapshot()`，无隐藏配置：

```
列数        = GRID_COLS      (默认 5)
行数        = GRID_ROWS     (默认 20 → 上限 100 张)
单格尺寸    = 800 × 533 px  (3:2，代码内常量 cell_w/cell_h)
排序        = 文件名时间戳倒序（最新在前）
范围        = SNAPSHOTS_DIR 下全部可解析快照，不限当天
横幅高度    = 110 px
```

每格绘制：左上姿态徽章（按姿态着色）+ 右上 `YYYY-MM-DD HH:MM:SS` 水印。

### 3.5 新增 HTTP 接口

`ui/app.py` 采用「一个函数一个路由」的扁平结构，新增时：

1. 在 `db.py` 补数据访问函数
2. 在 `app.py` 加 `@app.route("/api/xxx")`
3. 涉及配置写入时，同时更新 `DEFAULT_CONFIG`、`_apply_config_to_module()`、以及 `ui/templates/config.html` 的表单控件

---

## 🧪 4. 本地开发

### 4.1 免容器启动（Windows / macOS）

```bash
pip install -r requirements.txt -r ui/requirements-ui.txt
python dev_server.py            # 或 python -m ui.app
```

打开 http://127.0.0.1:8080 。

> ⚠️ **Windows 路径陷阱**：`OUTPUT_DIR` / `WORK_DIR` / `INPUT_SOURCES[].path` **必须是 Windows 绝对路径**（`c:/videocut/...`）。
> 写成 `/host/output` 会被 `pathlib.Path` 解析成当前盘符下的 `C:\host\output`，
> 且 `main.py` 中 `config.SNAPSHOTS_DIR = config.OUTPUT_DIR / "snapshots"` 会**强制覆盖**同名环境变量 —— 这是历史上最大的一个坑。

### 4.2 单次运行（不启服务）

```bash
python -m src.main \
  --input  c:/videocut/test_video/XiaomiCamera_00_<设备SN> \
  --output c:/videocut/devtest/output \
  --work   c:/videocut/devtest/work
```

或纯环境变量方式（`INPUT_DIR` / `OUTPUT_DIR` / `WORK_DIR`，扫描范围相关变量见 `src/config.py`）：

```bash
SCAN_TODAY_ONLY=false SCAN_PROCESS_HOURS=0 \
INPUT_DIR=c:/videocut/test_video/XiaomiCamera_00_<设备SN> \
OUTPUT_DIR=c:/videocut/devtest/output \
WORK_DIR=c:/videocut/devtest/work \
python -m src.main
```

### 4.3 单视频调试

```python
import sys; sys.path.insert(0, ".")
from pathlib import Path
from src import config, extract, detect, scanner

config.MODEL_DIR = Path("models")
dt = scanner.video_datetime(Path("test_video/XiaomiCamera_00_<设备SN>/xxx.mp4"), "auto")
print("parsed:", dt)

frames, _ = extract._extract_keyframes(Path(".../xxx.mp4"), Path("devtest/work/frames/dbg"))
print("keyframes:", len(frames))

res = detect.detect_pose(frames[:5])
print("detected:", len(res))
```

### 4.4 姿态规则单测

```python
from src.detect import classify_pose
kp = [[512, 100]]*17          # 构造 17 个 (x, y, conf)
kp[0]  = (256, 20)            # nose
kp[11] = (256, 90)            # L shoulder
kp[12] = (276, 90)            # R shoulder
kp[15] = (246, 200); kp[16] = (266, 200)   # ankles 靠拢 → 疑似 lie
print(classify_pose(kp, [0.9]*17))
```

---

## 🔍 5. 调试手册

### 5.1 日志过滤

```bash
docker compose logs -f videocut | grep -E '\[scan\]|\[extract\]|\[detect\]|\[retry\]|\[dedup\]|\[grid\]|\[main\]'
```

| 标记 | 出现位置 | 排查什么 |
|---|---|---|
| `[scan]` | scanner | 扫描到几个、跳过了几个、跳过原因 |
| `[extract][keyframe]` | extract | 关键帧数量、是否回退定距 |
| `[detect]` | detect | 批大小、单帧耗时、检出数 |
| `[retry]` | main | 小人物姿态重试是否命中 |
| `[dedup]` | dedup/main | 被 pHash 丢掉的帧 |
| `[grid]` | main | 拼接了几张、压缩后体积 |
| `[main]` | main | 任务汇总、耗时 |
| `[sched]` | ui/app | 调度生效值、定时任务安排 |

### 5.2 常见故障对照表

| 现象 | 首查 | 常见根因 |
|---|---|---|
| 大屏「人物 0」但磁盘有图 | `SNAPSHOTS_DIR` 与 UI `output_dir` 是否同一目录 | 路径写成 `/host/...` 被解析到错误盘符 |
| 快照一直不更新 | `docker compose logs \| grep '\[grid\]'` | `scan_process_hours` 把新视频全过滤掉了 |
| 大量 `pose-unknown` | 人物像素高度 | 画面太远/太暗；调 `DET_CONF` 无效，应看裁剪后最小边是否 < 200px |
| 同一场景反复出图 | `SIMILARITY_THRESHOLD` | 调低到 3；或 `DET_INTERVAL` 太大 |
| 漏掉明显有人时段 | 看 `[scan]` 跳过原因 | `SCAN_TODAY_ONLY` 与文件时间戳解析不一致 |
| 任务卡在「运行中」 | `/api/tasks` 的 `max_task_hours` | 看门狗 60s 一轮，超时会强杀并标记失败 |
| NAS 负载高但 GPU 0% | 正常 | OpenVINO CPU 推理；负载来自推理不是抽帧 |

### 5.3 数据重置

```bash
# 只清去重/事件状态，保留已产出的图片
rm -f /path/to/work/videocut.db*

# 让所有视频重新被处理（清 L2 视频去重）
sqlite3 /path/to/work/videocut.db "DELETE FROM processed_videos;"

# 只重跑网格拼接
python -c "import json,sys; sys.path.insert(0,'.'); from pathlib import Path; \
from src import config, main; \
c=json.load(open('devtest/work/config.json')); \
config.OUTPUT_DIR=Path(c['output_dir']); config.SNAPSHOTS_DIR=config.OUTPUT_DIR/'snapshots'; \
main._generate_grid_snapshot()"
```

---

## 🔐 6. 安全与隐私约束

代码里已经内置的几条硬性设计：

1. **原子写入**：`utils.atomic_write_bytes()` / `atomic_save_image()` 用「临时文件 → `flush` → `os.fsync` → `os.replace`」，避免 NAS 上写到一半被读到坏图。
2. **文件名白名单解析**：`_parse_snap_name()` 只接受能被时间戳规则解析的快照名，拼不进路径。
3. **`/preview/` 与 `/api/video`** 只服务 `OUTPUT_DIR` 内的文件。
4. **快照裁剪而非全帧**：`main._crop_to_subject()` 按检测框裁剪并按 3:2 统一，既减小体积也减少画面里的无关信息。

> 提 Issue / PR 时**不要附带真实监控画面、设备 ID / SN、本机绝对路径**。
> 仓库 `docs/images/` 下的配图由作者授权公开，仓库本身不含原始录像文件。

---

## 🚢 7. 发布流程

```bash
# 1. 版本号与变更记录
git tag -a v1.1.0 -m "feat: 小米 Format3 时间戳解析"
git push origin main --tags

# 2. Release
gh release create v1.1.0 \
  --title "v1.1.0" \
  --notes-file CHANGELOG.md

# 3. 用户升级
docker compose pull && docker compose up -d
```

`CHANGELOG.md` 按「新增 / 修复 / 变更 / 性能」四类分节，每条尽量带上影响面与迁移提示。

---

## ❓ 8. 二次开发常见问题

**Q：能不能不要 Flask，纯 CLI 跑？**
能。`src/` 不依赖 Flask，写个脚本调 `main.process(work_dir)` 即可，`ui/app.py` 只是它的一层壳。

**Q：为什么抽帧默认走关键帧而不是定距？**
监控录像的关键帧间隔通常几秒到几十秒，定距抽帧要解全片，代价高一个量级。实测 690s 4K HEVC：关键帧 117 帧 / 12.2s，定距 230 帧 / 69s。关键帧模式带 `showinfo` 拿真实时间戳，并有 300s 无输出看门狗，异常时自动回退定距。

**Q：为什么要做三层去重？**
L1 `DedupFilter`（pHash，进程内）挡住同一次运行里的重复画面；L2 `processed_videos`（文件名指纹入库）挡住跨次重跑；L3 `events` 表 + 文件名 glob 挡住重复事件写入。只做 L1 会在容器重启后失效。

**Q：改了 `src/config.py` 的默认值，UI 里没生效？**
UI 有自己的 `DEFAULT_CONFIG`（`ui/app.py`），且以 `WORK_DIR/config.json` 为准。改完记得同步 `ui/app.py` 的 `DEFAULT_CONFIG` 与 `_apply_config_to_module()`。

**Q：OpenVINO IR 在 Windows 上加载不到？**
`config.MODEL_DIR` 有个坑：Windows 上 `/app/models` 可能命中旧容器残留的 `C:\app\models`。代码已做保护 —— 仓库内 `models/` 含 INT8 IR 时优先用仓库目录。手动跑脚本时也建议先 `config.MODEL_DIR = Path("models")`。
