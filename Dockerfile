# Build with China-accessible mirrors so it does not depend on the NAS
# vendor registry (docker.fnnas.com). Adjust PYTHON_REGISTRY if needed.
# Pin to bookworm (Debian 12) because the Tsinghua apt mirror does not yet
# carry trixie (Debian 13) and would 403.
ARG PYTHON_REGISTRY=docker.1ms.run
FROM ${PYTHON_REGISTRY}/python:3.11-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    MODEL_DIR=/app/models \
    TZ=Asia/Shanghai \
    PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple \
    PIP_TRUSTED_HOST=mirrors.aliyun.com

# Use Aliyun (https) mirrors for apt (Debian); the Tsinghua http mirror
# returned 403 on this network, so we force https via Aliyun instead.
RUN find /etc/apt -name '*.list' -exec sed -i \
        -e 's|deb.debian.org/debian|mirrors.aliyun.com/debian|g' \
        -e 's|deb.debian.org/debian-security|mirrors.aliyun.com/debian-security|g' \
        -e 's|http://|https://|g' {} + 2>/dev/null; \
    find /etc/apt -name '*.sources' -exec sed -i \
        -e 's|deb.debian.org/debian|mirrors.aliyun.com/debian|g' \
        -e 's|deb.debian.org/debian-security|mirrors.aliyun.com/debian-security|g' \
        -e 's|http://|https://|g' {} + 2>/dev/null; \
    apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        curl \
        ca-certificates \
        tzdata \
        # 中文渲染兜底字体（拼接照片优先使用项目内置 src/fonts/wqy-zenhei.ttc，
        # 此包作为系统级备用；两者保证拼接照片中文不显示为小方块）
        fonts-wqy-zenhei \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install CPU-only torch + torchvision together (matching versions) first,
# avoiding the huge CUDA build. Must install both from the same index or
# torchvision's C++ ops (e.g. nms) won't match torch -> "operator
# torchvision::nms does not exist" at runtime.
RUN pip install --no-cache-dir torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt /app/requirements.txt
COPY ui/requirements-ui.txt /app/requirements-ui.txt
RUN pip install --no-cache-dir -r requirements.txt -r requirements-ui.txt
# ultralytics depends on the full opencv-python (needs GUI libs absent from
# slim images). Force the headless build to be the active cv2 to avoid
# ImportError at runtime.
RUN pip uninstall -y opencv-python 2>/dev/null; \
    pip install --no-cache-dir --force-reinstall --no-deps opencv-python-headless==4.9.0.80

# 统一模型：仅 yolov8n-pose.pt（检测+姿态一体，一次推理输出 person 框+17 关键点）。
# 运行时默认加载 INT8 OpenVINO IR（config.MODEL_PATH=/app/models/yolov8n-pose_int8_openvino_model）。
# 构建期从构建上下文 COPY 构建机已导出的 IR（python tools/export_model.py 提前导出）；
# 同时下载 .pt 作为干净检出时的兜底（detect.py 会按优先级回退）。
# 若构建上下文无 models/（干净检出），请先运行 python tools/export_model.py 再构建。
RUN mkdir -p /app/models \
    && fetch() { \
         raw="$1"; out="$2"; \
         for pre in "https://ghproxy.com/" "https://ghproxy.net/" ""; do \
           if curl -fsSL -o "$out" "${pre}${raw}"; then return 0; fi; \
         done; return 1; } \
    && fetch https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-pose.pt /app/models/yolov8n-pose.pt

# 构建机已导出的 OpenVINO IR（FP16 + 可选 INT8 混合精度）+ 本地 .pt
# （覆盖上面下载的同名 .pt，内容一致）。INT8 目录随 models/ 一并拷贝，
# 如需启用 INT8 推理，把 config.MODEL_PATH 指向 _int8_openvino_model 目录即可。
COPY models /app/models

COPY src /app/src
COPY ui /app/ui

EXPOSE 8080
ENTRYPOINT ["python", "-m", "ui.app"]
