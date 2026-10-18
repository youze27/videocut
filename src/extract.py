"""Extract frames from a video using FFmpeg (multi-threaded decode).

Decoding via FFmpeg is substantially faster than OpenCV's VideoCapture seek
and avoids keyframe seek overhead. Frames are written as JPGs to a temp dir
and returned as a sorted list of paths.
"""
import re
import subprocess
import threading
import time
from pathlib import Path

from . import config, utils

# 抽帧静默超时（秒）：ffmpeg 超过该时长没有任何输出（疑似卡死 / 网络挂载
# 无响应 / 损坏文件）即强制杀死并抛 TimeoutExpired，避免抽帧线程永久阻塞、
# 整条 pipeline 进度冻结。可调小以更快感知卡死，调大以容忍极慢的合法抽帧。
_EXTRACT_STALL_TIMEOUT = 300


def _extract_ffmpeg(cmd, progress_cb=None, video_name="", pts_sink=None, timeout=None):
    """运行 ffmpeg 抽帧，并实时上报已抽帧数。

    progress_cb(current)：每 ~1s 回调一次当前已输出帧数（ffmpeg 的 frame=）。
    pts_sink：可选 list；若提供，则把 stderr 里 showinfo 输出的每帧
    pts_time 按顺序追加进来（与 image2 写盘顺序一一对应）。
    timeout：静默超时（秒）。超过该时长 ffmpeg 无任何输出则强杀并抛
    subprocess.TimeoutExpired（与 utils.run 行为一致）。
    其它失败抛 subprocess.CalledProcessError（与 utils.run 行为一致）。
    """
    print("+ " + " ".join(str(c) for c in cmd))
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="utf-8", errors="replace")

    # 后台线程把 stderr 实时打到日志，保持原有流式日志行为
    def _pump_stderr():
        try:
            for line in proc.stderr:
                is_showinfo = "Parsed_showinfo" in line
                if is_showinfo:
                    m = re.search(r"pts_time:([0-9.]+)", line)
                    if m:
                        pts_sink.append(float(m.group(1)))
                # showinfo 输出行只用于收集 pts，不刷日志（每视频 ~200 行太吵）
                if not is_showinfo:
                    print(line, end="")
        except Exception:
            pass

    t = threading.Thread(target=_pump_stderr, daemon=True)
    t.start()

    last = time.time()
    last_activity = [time.time()]
    killed = [False]

    def _watchdog():
        # 静默超时看门狗：ffmpeg 卡死时不产生任何输出，readline 会永久阻塞，
        # 因此必须在独立线程里按"最后活动时间"主动强杀，而不是依赖 readline 返回。
        while proc.poll() is None:
            time.sleep(1.0)
            if timeout and time.time() - last_activity[0] > timeout:
                killed[0] = True
                try:
                    proc.kill()
                except Exception:
                    pass
                return

    if timeout:
        threading.Thread(target=_watchdog, daemon=True).start()

    current = 0
    try:
        while True:
            line = proc.stdout.readline()
            if not line:
                break
            last_activity[0] = time.time()
            line = line.strip()
            if line.startswith("frame="):
                try:
                    current = int(line.split("=", 1)[1].strip())
                except (ValueError, IndexError):
                    pass
            # 每个进度块结束时按 ~1s 节流上报一次，避免刷爆轮询接口
            if (line.startswith("progress=") and progress_cb
                    and time.time() - last >= 1.0):
                last = time.time()
                try:
                    progress_cb(current)
                except Exception:
                    pass
        proc.stdout.close()
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
        raise

    # 多给一点时间让 stderr 泵线程排空，确保 pts_sink 收集完整
    try:
        t.join(timeout=5)
    except Exception:
        pass
    rc = proc.wait()
    # 结束时补报一次最终帧数，确保界面停在准确数值
    if progress_cb:
        try:
            progress_cb(current)
        except Exception:
            pass
    if rc != 0:
        if killed[0]:
            raise subprocess.TimeoutExpired(cmd, timeout)
        print(f"[extract] ERROR: ffmpeg failed for {video_name} (exit={rc})")
        raise subprocess.CalledProcessError(rc, cmd)


def count_keyframes(video_path):
    """快速统计视频关键帧（I 帧）数量。ffprobe demux 级扫描，不解码。失败返回 0。"""
    cmd = [
        "ffprobe", "-v", "error",
        "-skip_frame", "nokey",
        "-select_streams", "v:0",
        "-show_entries", "frame=pts_time",
        "-of", "csv=p=0",
        str(video_path),
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True,
                             timeout=30, errors="replace").stdout
        return len([l for l in out.splitlines()
                    if l.strip() and not l.startswith("#")])
    except Exception as e:
        print(f"[extract] count_keyframes failed: {e}")
        return 0


def _extract_keyframes(video_path, out_dir, progress_cb=None, timeout=None):
    """关键帧快速抽帧：`-skip_frame nokey` 只解码 I 帧，不解码 P/B 帧。

    监控流编码器通常在场景变化/运动时才发关键帧（本机样本平均 ~6s 一帧），
    关键帧基本就是"事件所在帧"，比固定间隔抽帧快 ~5.7 倍且事件捕获更准
    （实测 690s 4K HEVC：关键帧 117 帧 12.2s vs 固定 3s 抽帧 230 帧 69s）。

    showinfo 过滤器把每帧 pts_time 打到 stderr，与 image2 写盘顺序一一对应，
    用来给每张帧恢复真实时间戳（关键帧间隔不规则，不能用 idx*interval）。

    返回 (frames, times)：frames 按时间升序，times[i] 为 frames[i] 的真实秒。
    失败/无关键帧时返回 ([], [])，由调用方回退到固定间隔抽帧。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # 清理上一轮残留帧
    for old in out_dir.glob("f_*.jpg"):
        try:
            old.unlink()
        except OSError:
            pass

    # ffprobe 预统计关键帧数，便于调试时判断"是否有可用关键帧"
    kf_count = count_keyframes(video_path)
    print(f"[extract][keyframe] {Path(video_path).name}: ffprobe 探测到 "
          f"{kf_count} 个关键帧，准备按关键帧模式抽帧")

    pattern = out_dir / "f_%06d.jpg"
    cmd = [
        "ffmpeg", "-y",
        "-skip_frame", "nokey",          # 输入侧只解码关键帧（I 帧）
        "-i", str(video_path),
        "-fps_mode", "passthrough",      # 每帧原样输出，不按固定 fps 平铺
        "-vf", "scale=1280:-2,showinfo", # 缩到 1280 宽 + 输出每帧 pts_time
        "-q:v", "3",
        "-progress", "pipe:1", "-nostats",
        str(pattern),
    ]
    print(f"[extract][keyframe] CMD: {' '.join(str(c) for c in cmd)}")
    pts = []
    try:
        _extract_ffmpeg(cmd, progress_cb=progress_cb,
                        video_name=Path(video_path).name, pts_sink=pts,
                        timeout=timeout)
    except Exception as e:
        print(f"[extract][keyframe] 抽帧异常，回退到固定间隔模式: {type(e).__name__}: {e}")
        return [], []

    frames = sorted(out_dir.glob("f_*.jpg"))
    if not frames:
        print(f"[extract][keyframe] 未写出任何帧文件，回退到固定间隔模式 "
              f"(ffprobe 曾报告 {kf_count} 个关键帧，可能是解码器未实际输出)")
        return [], []
    if not pts:
        print(f"[extract][keyframe] 写出 {len(frames)} 帧但未能从 showinfo 解析到 "
              f"pts_time，回退到固定间隔模式（时间轴不可用）")
        return [], []

    # pts 与帧数按较短者截断，保证一一对应
    n = min(len(frames), len(pts))
    frames, times = frames[:n], pts[:n]
    # 防御：pts 应单调递增；乱序说明异常，交给调用方回退
    if any(b < a for a, b in zip(times, times[1:])):
        print(f"[extract][keyframe] pts 非单调递增（{times[:3]}...{times[-3:]}），"
              f"回退到固定间隔模式")
        return [], []
    print(f"[extract][keyframe] 关键帧模式成功：实际抽出 {len(frames)} 帧，"
          f"pts 范围 {times[0]:.1f}s ~ {times[-1]:.1f}s")
    return frames, times


def extract_frames(video_path, interval_sec=None, out_dir=None, max_frames=None,
                   progress_cb=None, timeout=None):
    """抽帧：默认关键帧快速抽帧，失败/无关键帧时回退固定间隔抽帧。

    progress_cb(current)：可选，抽帧过程中每 ~1s 上报一次当前已抽帧数。
    timeout：抽帧静默超时（秒），透传给 ffmpeg 调用，防止卡死。
    Returns (frames, times)：frames 为按时间升序的 JPG 路径列表（可能为空）；
    times[i] 为 frames[i] 对应的视频时间秒（关键帧模式为真实 pts，
    回退模式为 idx*interval）。"""
    interval_sec = interval_sec if interval_sec is not None else config.DET_INTERVAL
    interval_sec = max(interval_sec, 0.1)
    out_dir = Path(out_dir) if out_dir else Path(config.WORK_DIR) / "frames"
    out_dir.mkdir(parents=True, exist_ok=True)
    if timeout is None:
        timeout = _EXTRACT_STALL_TIMEOUT

    t0 = time.time()

    # 记录视频文件大小，帮助诊断大文件
    try:
        fsize = Path(video_path).stat().st_size
        fsize_str = f"{fsize / 1024 / 1024:.1f}MB" if fsize > 1024 * 1024 else f"{fsize / 1024:.0f}KB"
    except Exception:
        fsize_str = "?"
    print(f"[extract] video={Path(video_path).name} ({fsize_str}), "
          f"interval={interval_sec:.1f}s (fallback), out={out_dir.name}")

    # ===== 主路径：关键帧快速抽帧（只解码 I 帧，监控场景快 ~5.7 倍）=====
    print(f"[extract] >>> 尝试关键帧模式抽帧（KEYFRAME MODE）for {Path(video_path).name}")
    frames, times = _extract_keyframes(video_path, out_dir, progress_cb,
                                        timeout=timeout)
    if frames:
        if max_frames and len(frames) > max_frames:
            frames, times = frames[:max_frames], times[:max_frames]
        elapsed = time.time() - t0
        print(f"[extract] <<< 关键帧模式成功：抽出 {len(frames)} 帧，"
              f"用时 {elapsed:.1f}s（{len(frames) / max(elapsed, 0.01):.0f} fps）")
        return frames, times

    # ===== 回退：固定间隔抽帧（关键帧缺失/失败时）=====
    print(f"[extract] >>> 回退到固定间隔模式（INTERVAL MODE），fps={1.0 / interval_sec:.2f} "
          f"for {Path(video_path).name}")
    # 清理上一轮残留帧
    old_count = 0
    for old in out_dir.glob("f_*.jpg"):
        try:
            old.unlink()
            old_count += 1
        except OSError:
            pass
    if old_count:
        print(f"[extract] cleaned {old_count} old frame(s) from {out_dir.name}")

    pattern = out_dir / "f_%06d.jpg"
    fps = 1.0 / interval_sec
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        # fps 先抽帧，scale 再缩小到 1280 宽：检测输入 640、快照 800 宽，
        # 4K 原分辨率纯属浪费（写盘/读图/缩放慢 9 倍）。1280 宽对下游足够。
        "-vf", f"fps={fps},scale=1280:-2", "-q:v", "3",
        # -progress pipe:1 输出机器可读进度（含 frame= 已抽帧数），
        # 供前端实时展示“抽到第几帧 / 已抽多少帧”；-nostats 关掉 stderr 刷屏
        "-progress", "pipe:1", "-nostats",
        str(pattern),
    ]
    try:
        _extract_ffmpeg(cmd, progress_cb=progress_cb,
                        video_name=Path(video_path).name, timeout=timeout)
    except Exception as e:
        # 抽帧彻底失败（含静默超时强杀）：返回空，由调用方跳过该视频，
        # 避免异常上抛导致整个任务/线程崩溃或卡死。
        print(f"[extract] ERROR: ffmpeg failed for {video_path}: {e}")
        return [], []

    frames = sorted(out_dir.glob("f_*.jpg"))
    if max_frames and len(frames) > max_frames:
        print(f"[extract] truncating {len(frames)} -> {max_frames} frames (limit)")
        frames = frames[:max_frames]
    times = [round(i * interval_sec, 2) for i in range(len(frames))]

    elapsed = time.time() - t0
    print(f"[extract] extracted {len(frames)} frame(s) in {elapsed:.1f}s "
          f"({len(frames) / max(elapsed, 0.01):.0f} fps)")
    return frames, times


def extract_frames_at_times(video_path, times, out_dir, timeout=120, progress_cb=None):
    """在指定时间点各抽取一帧（单次解码窗口，避免多次 -ss 慢速 seek）。

    times: [float, ...] 秒列表（姿态重试场景为 ±6s 内少量时间点）
    progress_cb(current, total=None)：抽帧过程中的进度回调（用于姿态重试时
    让前端进度条继续走动，避免"一直不走进度"）。
    返回 [(time_sec, frame_path), ...]：每个目标时间点映射到窗口内最近的
    0.5s 网格帧（对姿态重试足够精确）；窗口抽取失败/超时时回退到逐个
    -ss 抽取（带超时），两者都不成功的帧会缺失。

    背景：对慢速 NAS / 高码率 4K HEVC，逐个 `-ss -i` 需要各自从最近关键帧
    解码，多次 seek 极易卡死（无超时保护时会无限挂起）。改为单次 seek 到
    窗口起点、fps=2 一次解码整个 ±7s 窗口，全程只解码一次，快且可控。
    """
    if not times:
        return []
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t_min, t_max = min(times), max(times)
    # 窗口前后各留 1s 余量；fps=2 → 0.5s 一帧，覆盖所有目标时间点
    win_start = max(0.0, t_min - 1.0)
    win_dur = (t_max + 1.0) - win_start
    step = 0.5

    print(f"[extract][retry] 目标时间点={[round(t,1) for t in times]}，"
          f"窗口=[{win_start:.1f}s, {win_start+win_dur:.1f}s]，timeout={timeout}s")

    # 清理上一轮残留的重试帧，避免误用旧文件
    for old in out_dir.glob("r_*.jpg"):
        try:
            old.unlink()
        except OSError:
            pass

    pattern = out_dir / "r_%05d.jpg"
    cmd = [
        "ffmpeg", "-y", "-ss", f"{win_start:.2f}",
        "-i", str(video_path),
        "-t", f"{win_dur:.2f}",
        # 单次解码窗口：0.5s 一帧，缩到 1280 宽（与主抽帧一致）
        "-vf", f"fps={1.0 / step},scale=1280:-2", "-q:v", "3",
        str(pattern),
    ]
    try:
        _extract_ffmpeg(cmd, progress_cb=progress_cb,
                        video_name=Path(video_path).name, timeout=timeout)
    except subprocess.TimeoutExpired:
        print(f"[extract][retry] 窗口抽帧超时（{timeout}s），"
              f"回退到逐帧 -ss 抽取")
        return _extract_frames_at_times_seek(video_path, times, out_dir,
                                              timeout=timeout, progress_cb=progress_cb)
    except Exception as e:
        print(f"[extract][retry] 窗口抽帧失败（{type(e).__name__}: {e}），"
              f"回退到逐帧 -ss 抽取")
        return _extract_frames_at_times_seek(video_path, times, out_dir,
                                              timeout=timeout, progress_cb=progress_cb)

    frames = sorted(out_dir.glob("r_*.jpg"))
    if not frames:
        print(f"[extract][retry] 窗口抽帧未产出任何帧，回退到逐帧 -ss 抽取")
        return _extract_frames_at_times_seek(video_path, times, out_dir,
                                              timeout=timeout, progress_cb=progress_cb)

    results = []
    for t in times:
        # 第 i 帧（1 基）时间 ≈ win_start + (i-1)*step
        idx = int(round((t - win_start) / step))  # 0 基
        if 0 <= idx < len(frames):
            results.append((round(win_start + idx * step, 2), frames[idx]))
    print(f"[extract][retry] 窗口抽帧成功：{len(frames)} 帧，命中 {len(results)}/{len(times)} 个目标点")
    return results


def _extract_frames_at_times_seek(video_path, times, out_dir, timeout=60, progress_cb=None):
    """逐时间点 -ss 抽取（回退方案，带超时防止 ffmpeg 挂起卡死）。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    total = len([t for t in times if t >= 0])
    done = 0
    for t in times:
        if t < 0:
            continue
        out_path = out_dir / f"t_{int(t * 1000):08d}.jpg"
        cmd = [
            "ffmpeg", "-y", "-ss", f"{t:.2f}",
            "-i", str(video_path),
            "-frames:v", "1", "-q:v", "3", str(out_path),
        ]
        try:
            _extract_ffmpeg(cmd, progress_cb=None,
                            video_name=Path(video_path).name, timeout=timeout)
            if out_path.exists():
                results.append((t, out_path))
        except subprocess.TimeoutExpired:
            print(f"[extract][retry] 逐帧 {t:.1f}s 超时（{timeout}s），跳过")
        except Exception as e:
            print(f"[extract][retry] 逐帧 {t:.1f}s 失败（{type(e).__name__}: {e}），跳过")
        done += 1
        if progress_cb:
            try:
                progress_cb(done, total)
            except Exception:
                pass
    print(f"[extract][retry] 逐帧抽取完成：{len(results)}/{total} 个目标点成功")
    return results
