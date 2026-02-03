@echo off
REM TerraAgent - Flood Analysis Launcher
REM Double-click this file to start the application

echo ============================================
echo   TerraAgent - Flood Analysis Tool
echo ============================================
echo.

REM Jump to repo root (folder of this script)
cd /d "%~dp0"

REM Locate a Python executable (prefer local venv)
set "PY_EXE="
if exist ".venv\Scripts\python.exe" set "PY_EXE=.venv\Scripts\python.exe"
if "%PY_EXE%"=="" if exist "venv\Scripts\python.exe" set "PY_EXE=venv\Scripts\python.exe"
if "%PY_EXE%"=="" (
    for %%P in (python.exe) do where %%P >nul 2>nul && set "PY_EXE=python"
)

if "%PY_EXE%"=="" (
    echo [ERROR] Python not found. Please install Python or create a venv.
    pause
    exit /b 1
)

echo [INFO] Using Python: %PY_EXE%

REM Verify Streamlit is installed
%PY_EXE% -m streamlit --version >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Streamlit is not installed in this environment.
    echo Install with: %PY_EXE% -m pip install -r requirements.txt
    pause
    exit /b 1
)

echo Starting Flood Analysis Interface...
echo Press Ctrl+C to stop the server.
echo.

%PY_EXE% -m streamlit run generated_flood_app.py --server.port 8501 --server.headless true --browser.gatherUsageStats false
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Flood app exited with a non-zero status.
    pause
)
