@echo off
chcp 65001 >nul
REM ============================================================
REM NAS 智慧家视觉检测大屏 - 无人值守自动全屏启动脚本
REM ============================================================
REM 使用 Edge 的 --kiosk 模式启动，启动即全屏（比 F11 更彻底，
REM 无地址栏无标签栏，且无法用 Esc 退出，专为大屏展示设计）。
REM
REM 用法：
REM   1. 双击本脚本，或将本脚本加入"开机启动"实现真正无人值守
REM   2. 退出全屏：Alt+F4 关闭浏览器
REM ============================================================

set URL=http://127.0.0.1:8080/

REM 优先用 Microsoft Edge（Windows 10/11 自带），其次 Chrome
set "EDGE=%ProgramFiles%\Microsoft\Edge\Application\msedge.exe"
if not exist "%EDGE%" set "EDGE=%ProgramFiles(x86)%\Microsoft\Edge\Application\msedge.exe"
set "CHROME=%ProgramFiles%\Google\Chrome\Application\chrome.exe"
if not exist "%CHROME%" set "CHROME=%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe"

if exist "%EDGE%" (
    set "BROWSER=%EDGE%"
) else if exist "%CHROME%" (
    set "BROWSER=%CHROME%"
) else (
    echo [错误] 未找到 Edge 或 Chrome，请先安装其一。
    pause
    exit /b 1
)

REM --kiosk: 全屏模式（启动即全屏，无任何浏览器 UI）
REM --noerrdialogs: 屏蔽错误弹窗
REM --disable-session-crashed-bubble: 不显示"恢复上次会话"提示
REM --disable-restore-tab-state: 不恢复标签页状态
start "" "%BROWSER%" --kiosk --noerrdialogs --disable-session-crashed-bubble --disable-restore-tab-state "%URL%"
