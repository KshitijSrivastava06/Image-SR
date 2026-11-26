@echo off
setlocal
cd /d "%~dp0"

set "IMAGESR_PYTHON=python"
if exist ".venv\Scripts\python.exe" (
    set "IMAGESR_PYTHON=.venv\Scripts\python.exe"
) else (
    where python >nul 2>&1
    if errorlevel 1 set "IMAGESR_PYTHON=py"
)

"%IMAGESR_PYTHON%" run_web.py --open-browser %*
set "IMAGESR_EXIT=%ERRORLEVEL%"
if not "%IMAGESR_EXIT%"=="0" (
    echo.
    echo ImageSR Studio could not start. Check the error above.
    echo If Python or project dependencies are missing, see the setup in README.md.
    pause
)
exit /b %IMAGESR_EXIT%
