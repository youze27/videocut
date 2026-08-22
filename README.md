# videocut

一个带 Web UI 的 Docker 工具：扫描视频文件夹，找出**最近 N 小时**内录制的视频，用 **YOLOv8** 检测其中的**人 / 猫 / 狗**并识别**人物姿态**，为每个检测到的事件保存一张**围绕主体裁剪到 3:2 的 JPG 快照**。所有快照在 `index.json` 中记录源视频路径与起始时间，**NAS 监控大屏**按时间/对象/姿态/关键词筛选展示，点击图片直接跳转到源视频对应位置播放。已处理过的视频会被记录，后续只处理新增文件。

## 处理流水线

1. **扫描** – 递归遍历输入文件夹（支持多个摄像头子文件夹，任意 FFmpeg 可解码格式）。视频日期优先取自**文件名时间戳**（兼容监控摄像头：小米 / 海康 / 大华 / 通用，`auto` 模式自动识别），无法解析时回退到文件系统 mtime。默认 `SCAN_HOURS=2`，只处理最近 2 小时的视频。
2. **动态抽帧** – 跳过分段，直接在整个视频上用 **FFmpeg** 一次性抽帧（`-vf fps=N`）。抽帧间隔根据视频时长**动态调整**：小视频（<2分钟，如每小时1分钟的摄像头片段）用 2s 间隔；中等视频（2-10分钟）用 5s；大视频（>10分钟，如 128M 连续录像）用 10s。单次 ffmpeg 调用 + 最多 120 帧。
3. **检测** – **YOLOv8**（Ultralytics）批量检测 person/cat/dog。设置 `USE_POSE=true` 使用 `yolov8n-pose.pt` 模型，实现姿态感知检测（fall/stand/sit/squat/unknown）。
4. **去重** – 时间重叠（同一视频中相邻帧）+ 感知哈希视觉相似度双重去重。
5. **快照** – 每个不重复的检测事件，从抽帧结果中取出**置信度最高的一帧**，围绕检测框裁剪到统一 **3:2 比例**并缩放到固定尺寸（默认 720×480），人物帧绘制 COCO-17 骨架，保存为 JPG。同时在 `index.json` 记录源视频路径与帧时间戳。

### 已处理文件记录

流水线在 `<WORK_DIR>/processed.json` 中以 `path|size|mtime` 为 key 记录已处理视频。后续运行会跳过已处理文件，只处理新增/变化的视频。文件被替换（size 或 mtime 变化）后会重新处理。

## 构建与运行（Web UI）

```bash
docker compose build
docker compose up -d
```

然后打开 **http://localhost:8080**。

容器内部同时提供 UI 和每日定时调度。你需要**把宿主机目录挂载为 `/host`**（修改 `docker-compose.yml` 左侧），然后在 UI 中指定 `/host` 内的输入/输出子目录（例如 `/host/Videos`、`/host/output`）。"浏览"助手可列出某个路径下的子目录。

```bash
# 直接挂载宿主目录的独立运行方式：
docker run --rm -p 8080:8080 \
  -v "D:\Videos:/host" \
  -v "$PWD/work:/work" \
  videocut:latest
```

### UI 提供的功能

- **配置表单** – 输入/输出/工作目录、片段时长、扫描时间窗口、检测置信度、抽帧间隔、日期来源与摄像头品牌、姿态开关、去重阈值、快照比例/宽度/边距、每日自动运行开关及时间选择。
- **立即运行** – 立刻执行流水线（先保存当前配置）。
- **实时日志 + 任务历史** – 每次运行都记录状态、时间戳、产出文件及可查看的日志（"每日任务情况"）。运行中的日志实时刷新。
- **快照预览** – 生成的 JPG 快照直接在页面内展示。

每日输出（位于所选输出目录的 `snapshots/` 子目录）：

- `<YYYYMMDD_HHMMSS>_<obj>_pose-<pose>_<rand>.jpg` – 检测事件快照（人物绘制骨架）
- `index.json` – 每张快照对应的源视频路径与起始时间，供大屏跳转播放

`TZ`（默认 `Asia/Shanghai`）同时控制调度时间与时间戳判定。任务历史与配置持久化在 `WORK_DIR`（`/work`）下。

### 一次性 / 无 UI 运行

不开 Web 服务也能跑引擎：

```bash
docker run --rm \
  -v "D:\Videos:/host" -v "$PWD/work:/work" videocut:latest \
  python -m src.main --input /host --output /host/output --work /work
```

## 配置（环境变量）

| 变量 | 默认值 | 含义 |
|----------|---------|---------|
| `TZ` | `Asia/Shanghai` | 调度与时区判定 |
| `WORK_DIR` | `/work` | 临时空间 + 持久化配置/任务/已处理记录 |
| `SCAN_HOURS` | `2` | 只处理最近 N 小时的视频（0=按日期过滤） |
| `DATE_TOLERANCE_DAYS` | `0` | SCAN_HOURS=0 时允许往前多少天 |
| `DATE_SOURCE` | `either` | `filename` / `mtime` / `either` |
| `CAMERA_BRAND` | `auto` | `auto` / `xiaomi` / `hikvision` / `dahua` / `generic` |
| `DET_CONF` | `0.5` | 人体检测置信度 |
| `PET_CONF` | `0.5` | 猫/狗检测置信度 |
| `DET_INTERVAL` | `2.0` | 抽帧基础间隔（秒），大视频自动放大到 5-10s |
| `USE_POSE` | `true` | 使用 `yolov8n-pose.pt` 做姿态感知检测 |
| `MODEL_PATH` | `/app/models/yolov8n.pt` | YOLOv8 检测权重 |
| `POSE_MODEL` | `/app/models/yolov8n-pose.pt` | YOLOv8-pose 权重 |
| `MODEL_DIR` | `/app/models` | 构建时下载权重的目录 |
| `SIMILARITY_THRESHOLD` | `10` | 感知哈希汉明距离去重阈值 |
| `SNAPSHOTS_DIR` | `<OUTPUT>/snapshots` | 事件快照 JPG 输出目录 |
| `SNAPSHOT_RATIO` | `3:2` | 事件快照裁剪宽高比（宽:高） |
| `SNAPSHOT_WIDTH` | `720` | 事件快照输出宽度（px），高度按比例计算 |
| `SNAPSHOT_PADDING` | `0.25` | 裁剪区域相对检测框较大边的扩展比例 |

> 目录与大多数设置都在 UI 中配置；上表的环境变量作为默认值。所有 UI 设置会保存到 `<WORK_DIR>/config.json`。

## 备注 / 限制

- **多摄像头支持**：扫描递归遍历输入目录的所有子文件夹，支持多个摄像头文件夹的父级目录。`CAMERA_BRAND=auto` 自动识别小米/海康/大华/通用等文件名时间戳格式，混合品牌摄像头也能正确解析。
- **动态抽帧间隔**：根据视频时长自动调整抽帧密度，平衡检测覆盖度与处理速度。小视频（<2分钟）2s/帧、中等视频（2-10分钟）5s/帧、大视频（>10分钟，如 128M 连续录像）10s/帧。大视频不再分段，单次 ffmpeg 抽帧 + 批量检测，大幅减少 I/O 开销。
- 人体检测使用 **YOLOv8**（Ultralytics），仅用于决定保留/跳过帧并选取关键帧。**本项目不生成合并视频或拼图**，只输出按事件裁剪的 JPG 快照。`USE_POSE=true` 会切换为 `yolov8n-pose.pt` 模型，人物快照会绘制 **COCO-17 骨架**。
- 姿态标签由基于规则的轻量分类器基于 COCO-17 关键点得出（不依赖额外模型）。模糊姿态回退到 `unknown`；这是启发式判定，不是医疗级跌倒检测。
- 快照裁剪：围绕检测框居中裁剪到 3:2，主体靠近画面边缘时用黑边补齐比例，保证所有快照尺寸完全一致。
- 抽帧使用 **FFmpeg**（`-vf fps=N`），比逐帧 `cv2.VideoCapture` seek 在长录像上快得多。`ffprobe` 不可用时回退到 `cv2` 验证视频可读性。
- 镜像使用**国内可访问的镜像源**构建（docker.1ms.run 基础镜像、阿里云 apt + PyPI），YOLOv8 权重通过 GitHub 代理（ghproxy）下载，使 `docker build` 在 GFW 内也能成功。构建只需联网一次。

## NAS 监控事件大屏 (Event dashboard)

流水线为**大屏查看器**输出**按事件的关键帧 JPG**。每个包含人/猫/狗的保留片段会在 `SNAPSHOTS_DIR` 下产出一张围绕检测框裁剪到 3:2 的 JPG（人物绘制骨架），并在 `SNAPSHOTS_DIR/index.json` 中记录其**源视频路径与起始时间**：

```
<YYYYMMDD_HHMMSS>_<obj>_pose-<pose>_<rand>.jpg   # 裁剪到 3:2 的事件快照
index.json                                        # 源视频路径 + 起始时间
```

- `obj` ∈ `person` / `cat` / `dog`；`pose` ∈ `fall` / `stand` / `sit` / `squat` / `unknown` / `none`（猫/狗始终为 `none`）。
- 大屏是**主屏幕**，位于 **`/`**（端口 8080）。控制面板（运行流水线 / 配置 / 任务历史）从大屏上的 **⚙ 配置** 按钮打开（也可直接访问 **`/config`**）。大屏完全在客户端筛选（对象 / 姿态 / 时间 / 关键词）、图片懒加载、**点击图片在模态框中用 HTML5 video 播放源视频并自动定位到起始时间**（无需插件），并每 30s 自动刷新。对象标签与姿态标签使用两套独立的颜色系统。如需让大屏单独运行在 5000 端口：`python -m ui.web_server`。
