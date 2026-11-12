@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

if "%~1"=="" (
  echo Usage: drag a video file onto this bat, or run:  one_click_analyze.bat "D:/path/video.mp4"
  pause
  exit /b 1
)

set "VIDEO=%~1"
set "NAME=%~n1"
set "OUT=%~dp0..\outputs_upstream\%NAME%"

echo ============================================================
echo   Good-Badminton analysis (upstream pipeline)
echo   input : %VIDEO%
echo   output: %OUT%
echo ============================================================

set GB_AUTO_ACCEPT=1
"%~dp0..\.venv\Scripts\python.exe" -u "%~dp0main.py" --video-path "%VIDEO%" --display false --audio false --output-dir "%OUT%"

if errorlevel 1 (
  echo.
  echo [ERROR] analyze failed, see messages above.
  pause
  exit /b 1
)

echo.
echo Done. Opening output folder...
start "" "%OUT%"
exit /b 0
