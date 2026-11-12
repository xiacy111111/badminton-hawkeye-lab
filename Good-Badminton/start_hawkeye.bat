@echo off
setlocal
pushd "%~dp0"
set "PYW=%~dp0..\.venv\Scripts\pythonw.exe"
if not exist "%PYW%" goto fallback
start "" /D "%~dp0" "%PYW%" "%~dp0app_gui.py"
popd
exit /b 0
:fallback
echo [ERROR] venv pythonw not found: %PYW%
popd
pause
exit /b 1
