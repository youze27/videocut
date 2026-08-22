# NAS 智慧家视觉检测系统 — 使用文档

> 扫描 NAS 上的摄像头视频 → 抽帧 → YOLOv8 识别人/猫/狗等对象 + 姿态 → 自动生成快照 → 大屏轮播展示，全程自动化运行。

---

## 目录

1. [快速开始](#1-快速开始)
2. [部署与启动](#2-部署与启动)
3. [自动化运行机制](#3-自动化运行机制)
4. [大屏使用说明](#4-大屏使用说明)
5. [配置参数详解](#5-配置参数详解)
6. [API 接口参考](#6-api-接口参考)
7. [硬件性能与优化建议](#7-硬件性能与优化建议)
8. [目录结构与文件说明](#8-目录结构与文件说明)
9. [常见问题](#9-常见问题)

---

## 1. 快速开始

### 最简部署（Docker）

```bash
docker compose up -d
```

服务启动后：
- **大屏监控页**：`http://<NAS IP>:8080/`
- **配置页**：`http://<NAS IP>:8080/config`（也可从大屏右上角「⚙ 配置」按钮进入）

启动即全自动运行，无需人为操作：
1. 服务就绪后 3 秒自动触发首次扫描
2. 之后每 1 小时定时扫描一次
3. 检测到新视频会自动触发 pipeline
4. 每日凌晨 4 点自动清理过期临时帧

### 首次使用清单

1. 把摄像头录像目录挂载到容器 `/host`（已在 `docker-compose.yml` 配置）
2. 启动容器
3. 打开大屏页面，点击「⚙ 配置」确认「输入源」指向正确目录
4. 等待首次自动扫描完成，快照会自动出现在大屏轮播中

---

## 2. 部署与启动

### 2.1 docker-compose.yml 说明

```yaml
services:
  videocut:
    build: .
    restart: unless-stopped
    ports:
      - "8080:8080"
    volumes:
      - /vol2/1000/video_back:/host          # NAS 视频目录 → 容器 /host
      - /vol2/1000/video_back/.videocut_work:/work  # 工作目录（配置/日志/临时帧）
      - ./src:/app/src                       # 挂载源码，改代码免重建镜像
      - ./ui:/app/ui
    environment:
      TZ: Asia/Shanghai
      WORK_DIR: /work
```

**关键挂载点：**

| 容器路径 | 作用 | 修改建议 |
|---------|------|---------|
| `/host` | NAS 视频源（摄像头录像） | 改左边为你的实际视频目录 |
| `/work` | 工作目录：config.json、tasks.json、logs、frames | 建议放 SSD 或快速磁盘 |
| `/app/src` `/app/ui` | 源码挂载 | 修改代码后重启容器即可生效，**无需重建镜像** |

### 2.2 启动方式

```bash
# 后台启动
docker compose up -d

# 查看日志
docker compose logs -f

# 重启（改完代码后生效）
docker compose restart

# 重建镜像（仅 Dockerfile/依赖变更时需要）
docker compose up -d --build
```

### 2.3 访问入口

| URL | 说明 |
|-----|------|
| `http://<IP>:8080/` | 大屏监控主页（轮播展示快照） |
| `http://<IP>:8080/config` | 配置与任务管理页 |
| `http://<IP>:8080/api/...` | REST API（见第 6 节） |

---

## 3. 自动化运行机制

服务采用**多层自动化**，启动后完全无需人为干预：

```
容器启动
  ├─ reschedule() 注册定时任务
  ├─ 后台线程 _startup_autorun()
  │    └─ 延迟 3 秒 → 触发首次扫描（task_id 入队）
  │
  └─ APScheduler 后台调度
       ├─ scheduled（定时检测）  每 N 小时运行一次
       ├─ watch（目录监听）      每 N 分钟检查视频列表变化，有变化则触发
       └─ cleanup（每日清理）    每天 04:00 清理过期临时帧
```

### 3.1 任务队列

所有检测任务通过 `_task_queue` 串行执行，**防止并发冲突**：
- 任意时刻最多 1 个任务在跑
- 队列最多缓存 5 个待处理任务，超出则拒绝
- 触发源（定时/watch/手动/启动）统一入队

### 3.2 跳过已处理视频

每个视频用 `path|size|mtime` 作为唯一 key 记录到 `processed.json`，重复扫描会自动跳过，不会重复处理。

> 清空 `processed.json` 可强制重处理所有视频；跨运行去重逻辑（同时间戳+同对象+同姿态的快照已存在则跳过）保证不会产生重复快照。

### 3.3 Pipeline 流程（每个视频）

```
视频文件
  ↓ extract_frames (ffmpeg 抽帧，每 DET_INTERVAL 秒一帧，最多 120 帧)
  ↓ detect_objects (YOLOv8 det 模型识别对象)
  ↓ person + pose=unknown? → pose 模型再识别 + 小人物裁剪重试
  ↓ pose 仍 unknown? → _retry_pose 抽附近 ±6s 共 6 帧重试
  ↓ pHash 视觉去重（Hamming 距离 ≤ 阈值 视为重复，跳过）
  ↓ _emit_event: 裁剪到 3:2 + 画骨架 + 保存 JPG + 记录 index.json
  ↓ 前端大屏检测到新事件 → 自动截图大屏 → 保存 grid_latest.jpg
```

---

## 4. 大屏使用说明

### 4.1 主页布局（`/`）

```
┌─────────────────────────────────────────────────────┐
│ 标题  统计(总数/人/猫/狗)  时钟  [手动刷新][清空筛选] │  顶部栏
│       [自动刷新:30s][📹无对象视频][⚙配置]            │
├─────────────────────────────────────────────────────┤
│ 对象:[全部▼] 姿态:[全部▼] 时间:[全部▼] 布局:[4x5▼]  │  筛选栏
├─────────────────────────────────────────────────────┤
│                                                       │
│         轮播展示区（4x5 = 每页 20 张）                │  主区域
│                                                       │
├─────────────────────────────────────────────────────┤
│ ‹  第1/5页  ›    [▬▬▬▬▬▬▬░░░] 5s进度条              │  控制条（全屏时鼠标靠近底部显示）
└─────────────────────────────────────────────────────┘
```

### 4.2 交互说明

| 操作 | 效果 |
|------|------|
| **5 秒无操作** | 自动进入全屏模式（kiosk），只保留顶部栏 + 轮播 |
| **鼠标靠近底部** | 显示轮播控制条（翻页/进度） |
| **鼠标移出底部** | 控制条隐藏 |
| **点击 ‹ ›** | 手动翻页（会重置 5 秒计时） |
| **全屏时点右上角「退出」** | 退出全屏 |
| **对象/姿态下拉** | 多选筛选，可同时选多个对象或姿态 |
| **时间下拉** | 单选：全部 / 当天 / 3天 / 7天 / 30天 |
| **布局下拉** | 单选 1x1 ~ 6x6，控制每页展示张数 |
| **自动刷新** | 每 30 秒拉取最新事件，检测到新事件自动截图 |

### 4.3 展示数量限制

- 大屏最多展示**最新 360 张**快照（`MAX_DISPLAY = 360`，[dashboard.html](file:///c:/videocut/ui/templates/dashboard.html)）
- 超过 360 个事件时只展示最新 360 张，计数器会提示「仅展示最新 360」
- 顶部统计数字仍显示真实总数，不受限制

### 4.4 自动截图大屏

- 前端每 30 秒自动刷新事件
- 检测到事件总数增加（说明 pipeline 跑完产生新数据）→ 2 秒后自动用 `html2canvas` 截取整个大屏 → 上传保存为 `grid_latest.jpg`
- 外部可通过 `http://<IP>:8080/api/grid_latest` 访问这张截图

### 4.5 快照排序

所有快照按**时间倒序**排列（最新的在前），见 [dashboard.html](file:///c:/videocut/ui/templates/dashboard.html) `applyFilter()` 中的 `filtered.sort((a, b) => (b.ts || 0) - (a.ts || 0))`。

---

## 5. 配置参数详解

配置存储在 `/work/config.json`，可通过 `/config` 页面或 `/api/config` 接口修改。所有参数都有默认值，开箱即用。

### 5.1 输入输出

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `input_sources` | `[{"path":"/host","brand":"auto"}]` | 输入源列表，支持多摄像头目录。每个源可独立指定品牌用于时间戳解析 |
| `output_dir` | `/host/output` | 输出目录，快照保存在其下 `snapshots/` 子目录 |
| `work_dir` | `/work` | 工作目录：config.json、tasks.json、logs/、frames/ |

**input_sources 格式：**
```json
[
  {"path": "/host/cam1", "brand": "xiaomi"},
  {"path": "/host/cam2", "brand": "hikvision"}
]
```

**camera_brand 取值：**
- `auto`：自动识别（默认）
- `xiaomi`：小米摄像头文件名时间戳格式
- `hikvision`：海康威视
- `dahua`：大华
- `generic`：通用格式

### 5.2 检测参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `det_conf` | `0.5` | 人物检测置信度阈值，越高越严格（误检少但可能漏检） |
| `det_interval` | `4.0` | **抽帧间隔（秒）**。每 N 秒抽 1 帧去识别。增大→负载降低但可能漏掉快速动作。**这是降负载的关键参数** |
| `similarity_threshold` | `5` | pHash 视觉去重 Hamming 距离阈值。≤此值视为重复帧跳过。监控场景背景固定，建议 5；场景变化大可调到 8-10 |
| `use_pose` | `true` | 是否启用姿态识别（yolov8n-pose）。关闭则只检测对象不识别姿态，负载减半 |
| `worker_threads` | `"auto"` | 多视频并行线程数。`auto`=按 CPU 核心数（最多4），或指定整数。**J1800 等弱 CPU 建议设 1** |

**det_interval 负载对照（2分钟视频）：**

| det_interval | 抽帧数 | 相对负载 | 适用场景 |
|--------------|--------|---------|---------|
| 2.0s | 60 帧 | 100% | 高灵敏度，快速动作捕捉 |
| 4.0s（默认） | 30 帧 | 50% | 监控场景平衡选择 |
| 6.0s | 20 帧 | 33% | 低负载，长时间静止场景 |
| 10.0s | 12 帧 | 20% | 极低负载，只看大致情况 |

### 5.3 快照输出

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `snapshot_ratio` | `"3:2"` | 快照裁剪比例（宽:高）。统一比例让大屏展示整齐 |
| `snapshot_width` | `720` | 快照输出宽度（px），高度按比例计算 |
| `snapshot_padding` | `0.25` | 裁剪区域相对检测框的扩展比例（0.25=扩25%边距，避免切掉肢体） |

**快照文件命名：**
```
{YYYYMMDD_HHMMSS}_{对象}_pose-{姿态}_{随机3位}.jpg
例: 20260822_102530_person_pose-stand_a3f.jpg
```

### 5.4 自动化调度

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `schedule_enabled` | `true` | 是否启用定时检测 |
| `schedule_interval_hours` | `1` | 定时检测间隔（小时）。每 N 小时自动运行一次 pipeline |
| `watch_enabled` | `true` | 是否启用目录监听。检查视频列表变化，有新视频自动触发 |
| `watch_interval` | `60` | watch 检查间隔（分钟）。每 N 分钟检查一次视频列表 |
| `cleanup_enabled` | `true` | 是否启用每日自动清理 |
| `cleanup_time` | `"04:00"` | 每日清理触发时间（HH:MM） |
| `frame_retention_days` | `30` | 临时帧保留天数。超过自动删除。**已输出的快照不会被删**。0=不清理 |

**自动化触发时机汇总：**

| 触发源 | 时机 | 说明 |
|--------|------|------|
| 启动自动扫描 | 服务就绪后 3 秒 | 首次启动立即跑一遍，处理已有视频 |
| 定时检测 | 每 `schedule_interval_hours` 小时 | 周期性扫描，跳过已处理视频 |
| 目录监听 | 每 `watch_interval` 分钟检查 | 视频列表有变化才触发，避免无意义重跑 |
| 每日清理 | 每天 `cleanup_time` | 只删临时帧，不删快照 |
| 手动触发 | `/config` 页点「运行」 | 立即入队执行 |

### 5.5 网格截图

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `grid_cols` | `4` | 网格截图列数（已弃用后端拼接，改由前端 html2canvas 截取大屏） |
| `grid_rows` | `3` | 网格截图行数 |

> 当前网格截图由前端大屏自动截取（pipeline 跑完产生新事件后触发），后端不再拼接。`grid_latest.jpg` 保存的是大屏完整截图。

---

## 6. API 接口参考

所有接口均在 `http://<IP>:8080` 下。

### 6.1 配置

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/config` | 获取当前配置 |
| POST | `/api/config` | 更新配置（传 JSON，部分字段即可） |

**示例 - 修改抽帧间隔：**
```bash
curl -X POST http://localhost:8080/api/config \
  -H "Content-Type: application/json" \
  -d '{"det_interval": 6.0}'
```

### 6.2 任务管理

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/run` | 触发一次检测任务，返回 task_id |
| POST | `/api/cancel` | 取消当前运行的任务 |
| GET | `/api/tasks` | 获取所有任务列表与队列状态 |
| GET | `/api/task/<task_id>` | 获取单个任务详情与日志 |
| GET | `/api/progress` | 获取当前任务进度 |
| GET | `/api/summary` | 获取当前任务各视频处理摘要 |

### 6.3 扫描与浏览

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/scan?input_dir=/host&brand=auto` | 扫描指定目录的视频列表 |
| GET | `/api/browse?path=/host` | 浏览目录结构（配置页目录选择器用） |

### 6.4 输出与预览

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/previews` | 获取所有快照列表 |
| GET | `/preview/snapshots/<filename>` | 预览指定快照 |
| GET | `/api/grid_latest` | 获取最新大屏截图 |
| GET | `/api/grid_info` | 获取网格截图配置与状态 |
| POST | `/api/grid_upload` | 上传大屏截图（前端自动调用） |

### 6.5 无对象视频管理

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/no_object_videos` | 列出未检测到任何对象的视频 |
| DELETE | `/api/no_object_videos` | 删除指定视频（传 `{"paths":[...]}`) |

### 6.6 视频播放

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/video?path=/host/xxx.mp4` | 流式播放视频（路径必须在输入源目录下，安全校验） |

### 6.7 事件数据

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/events` | 获取所有事件（大屏数据源，返回 events + stats） |

### 6.8 清理

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/cleanup` | 手动触发清理过期临时帧 |

---

## 7. 硬件性能与优化建议

### 7.1 负载来源分析

pipeline 各阶段开销（从高到低）：

| 阶段 | 负载 | 说明 |
|------|------|------|
| **YOLOv8 推理（识别）** | 🔴 极高 | 每帧跑 CNN，占总耗时 80-95% |
| 姿态重试（6帧重跑） | 🔴 极高 | person+unknown 时额外 6 次推理 |
| 小人物裁剪重试 | 🟠 中 | bbox<200px 时 pose 模型再跑一次 |
| ffmpeg 抽帧 | 🟢 低 | 高度优化的 C 代码，视频解码很快 |
| PIL 快照处理 | 🟢 低 | 普通图像裁剪缩放 |

**识别是负载元凶**，抽帧很轻。降负载的核心是减少推理次数。

### 7.2 GPU 说明

本系统基于 ultralytics YOLOv8，**仅支持 NVIDIA CUDA GPU**。

- **无 NVIDIA GPU**（如 Intel 核显、AMD）：自动使用 CPU 推理，GPU 显示 0% 是正常的
- **Dockerfile 默认装 CPU 版 PyTorch**：即使有 NVIDIA GPU 也用不了，需改 Dockerfile 装 CUDA 版 + docker-compose 加 GPU 映射

### 7.3 优化建议（按收益排序）

| 优化项 | 操作 | 效果 | 代价 |
|--------|------|------|------|
| **增大 det_interval** | 配置页改 4.0→6.0 | 帧数减半，负载减半 | 可能漏掉快速动作 |
| **关闭姿态重试** | 代码层（见下） | 砍掉 6 帧额外推理 | unknown 姿态增多 |
| **关闭 use_pose** | 配置页设 false | 推理量减半 | 不识别姿态 |
| **减小 imgsz** | 代码层 640→416 | 单帧计算量降 60% | 远距离小目标精度降 |
| **worker_threads=1** | 配置页设 1 | 避免弱 CPU 抢占 | 多视频串行处理 |
| **导出 OpenVINO 模型** | 需重建镜像 | Intel CPU 提速 1.5-2x | J1800 无 AVX2 收益打折 |

### 7.4 J1800 等弱 CPU 推荐配置

```json
{
  "det_interval": 6.0,
  "worker_threads": 1,
  "use_pose": true,
  "schedule_interval_hours": 2
}
```

### 7.5 如何从日志验证负载分布

服务日志中各阶段都有计时，可直接看占比：

```
[extract] video=xxx.mp4 (50.2MB), interval=4.0s, ...           ← 抽帧开始
[extract] extracted 30 frame(s) in 3.2s (9 fps)                 ← 抽帧耗时
[detect] starting: 30 frames, ...                               ← 识别开始
[detect] done in 45.3s: 8/30 frames had detections (...)        ← 识别耗时（占大头）
[main] xxx.mp4: 8 snapshot(s) saved, ... took 52.1s             ← 单视频总耗时
```

典型情况：**识别耗时占总耗时的 80-95%**，抽帧占 5-15%。

---

## 8. 目录结构与文件说明

```
c:\videocut\
├── docker-compose.yml          # 部署配置
├── Dockerfile                  # 镜像构建（CPU 版 PyTorch）
├── requirements.txt            # Python 依赖
├── src\                        # 核心 pipeline 代码
│   ├── config.py              # 所有配置参数（含默认值与环境变量）
│   ├── main.py                # pipeline 主流程：抽帧→检测→快照
│   ├── detect.py              # YOLOv8 检测与姿态识别
│   ├── extract.py             # ffmpeg 抽帧
│   ├── scanner.py             # 视频扫描与时间戳解析
│   ├── dedup.py               # pHash 视觉去重
│   └── utils.py               # 工具函数
├── ui\                         # Web UI
│   ├── app.py                 # Flask 服务 + APScheduler 调度
│   ├── web_server.py          # 大屏 dashboard 蓝图
│   └── templates\
│       ├── dashboard.html     # 大屏监控主页
│       └── index.html         # 配置与任务管理页
└── README.md                   # 本文档

容器内运行时目录：
/work/
├── config.json                 # 持久化配置
├── tasks.json                  # 任务历史记录
├── processed.json              # 已处理视频记录（key=path|size|mtime）
├── no_object_videos.json       # 未检测到对象的视频记录
├── logs/                       # 每个任务的完整日志
│   └── 20260822-102530-a3f.log
└── frames/                     # 临时抽帧文件（按 retention_days 自动清理）

/host/output/snapshots/         # 快照输出目录
├── 20260822_102530_person_pose-stand_a3f.jpg
├── index.json                  # 快照→源视频映射（大屏点击跳转播放用）
└── grid_latest.jpg             # 最新大屏截图（前端自动截取）
```

---

## 9. 常见问题

### Q1：GPU 使用率 0%？

**A：** 正常现象。本系统仅支持 NVIDIA CUDA，若你的 NAS 是 Intel/AMD 核显则无法用 GPU，自动回退 CPU 推理。详见 [第 7.2 节](#72-gpu-说明)。

### Q2：有些识别到的对象没有快照？

**A：** 通常是 pHash 视觉去重把相似画面过滤了。降低 `similarity_threshold`（默认 5）到 3 可减少误去重，但可能产生相似快照。监控场景背景固定，5 是平衡值。

### Q3：出现重复快照？

**A：** 跨运行去重逻辑会检查同时间戳+同对象+同姿态的快照是否已存在。若仍重复，检查 `snapshots/` 目录是否有手动复制的文件。

### Q4：服务启动后没有自动运行？

**A：** 检查：
1. `config.json` 中 `schedule_enabled` 和 `watch_enabled` 是否为 `true`
2. `input_sources` 路径是否正确且有视频文件
3. 查看 `/work/logs/` 下最新日志是否有报错

### Q5：如何强制重新处理所有视频？

**A：** 删除 `/work/processed.json`，下次扫描会重新处理所有视频。跨运行去重逻辑保证不会产生重复快照。

### Q6：大屏不显示图片？

**A：** 检查：
1. `/host/output/snapshots/` 下是否有 jpg 文件
2. `/api/events` 接口是否返回数据
3. 浏览器控制台（F12）是否有报错

### Q7：如何修改代码并生效？

**A：** 由于 `docker-compose.yml` 挂载了 `./src` 和 `./ui`，修改代码后只需：
```bash
docker compose restart
```
无需重建镜像。仅当修改 `Dockerfile` 或 `requirements.txt` 时才需要 `docker compose up -d --build`。

### Q8：姿态显示 unknown 怎么办？

**A：** unknown 表示关键点不足无法判断姿态，常见于：
- 人物太小（远处或局部入画）
- 遮挡严重
- 系统已内置「小人物裁剪+上采样重试」和「附近时间点抽帧重试」两次补偿机制，仍失败则标 unknown

可适当降低 `det_conf` 或减小 `det_interval` 增加捕捉到清晰姿态的机会。

### Q9：大屏截图（grid_latest.jpg）不更新？

**A：** 大屏截图由前端自动触发：检测到事件总数增加 → 2 秒后自动截图。若不更新：
1. 确认大屏页面在浏览器中保持打开
2. 检查「自动刷新」是否开启（按钮应显示绿色 `自动刷新: 30s`）
3. F12 控制台查看 `[screenshot]` 日志

---

## 附录：默认配置一览

```json
{
  "input_sources": [{"path": "/host", "brand": "auto"}],
  "output_dir": "/host/output",
  "work_dir": "/work",
  "det_conf": 0.5,
  "det_interval": 4.0,
  "date_source": "either",
  "camera_brand": "auto",
  "use_pose": true,
  "similarity_threshold": 5,
  "snapshot_ratio": "3:2",
  "snapshot_width": 720,
  "snapshot_padding": 0.25,
  "schedule_enabled": true,
  "schedule_interval_hours": 1,
  "watch_enabled": true,
  "watch_interval": 60,
  "worker_threads": "auto",
  "frame_retention_days": 30,
  "cleanup_enabled": true,
  "cleanup_time": "04:00",
  "grid_cols": 4,
  "grid_rows": 3
}
```

---

**文档版本：** 2026-08-22
**项目路径：** `c:\videocut\`
