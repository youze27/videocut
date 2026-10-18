#!/bin/bash
# ============================================================
#  NAS 视觉检测大屏 - 展示模式启动器（Linux）
#  用 Chrome/Edge 的 --kiosk 参数启动，实现真正的 F11 全屏效果
#
#  用法：bash start_display.sh
#  退出：Alt+F4
# ============================================================

URL="http://127.0.0.1:8080/dashboard"
FLAGS="--kiosk --fullscreen --noerrdialogs --disable-session-crashed-bubble --disable-restore-tab-state"

# 按优先级查找浏览器
for BIN in \
    /usr/bin/microsoft-edge \
    /usr/bin/google-chrome \
    /usr/bin/chromium-browser \
    /usr/bin/chromium \
    /opt/microsoft/msedge/msedge \
    /opt/google/chrome/chrome; do
    if [ -x "$BIN" ]; then
        exec "$BIN" $FLAGS "$URL"
    fi
done

echo "未找到 Chrome/Edge/Chromium，请先安装浏览器。"
exit 1
