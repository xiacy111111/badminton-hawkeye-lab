@echo off
cd /d "%~dp0"
start "HawkEye" "%~dp0..\.venv\Scripts\pythonw.exe" "%~dp0app_gui.py"
if errorlevel 1 start "HawkEye" "%~dp0..\.venv\Scripts\python.exe" "%~dp0app_gui.py"
if errorlevel 1 pause
