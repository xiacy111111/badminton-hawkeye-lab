@echo off
chcp 65001 >nul
cd /d "%~dp0"
title 羽毛球鹰眼分析台
if exist "%~dp0..\.venv\Scripts\pythonw.exe" (
    start "" "%~dp0..\.venv\Scripts\pythonw.exe" app_gui.py
) else (
    echo 未找到虚拟环境，请先运行:
    echo   python -m venv ..\.venv
    echo   ..\.venv\Scripts\pip install numpy opencv-python pillow windnd
    pause
)
