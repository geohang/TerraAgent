@echo off
REM ============================================================
REM TerraAgent - One-click launcher (Windows)
REM Double-click to create/activate the conda env and open the app
REM ============================================================

echo.
echo  ========================================
echo   TerraAgent - Streamlit Platform
echo  ========================================
echo.

REM Jump to repo root (folder of this script)
cd /d "%~dp0"

REM Ensure conda is available (try PATH first, then common locations)
set "CONDA_EXE="
for %%P in (conda.exe) do where %%P >nul 2>nul && set "CONDA_EXE=conda.exe"
if "%CONDA_EXE%"=="" (
    for %%C in (
        "C:\ProgramData\Anaconda3\Scripts\conda.exe"
        "%USERPROFILE%\anaconda3\Scripts\conda.exe"
        "%USERPROFILE%\miniconda3\Scripts\conda.exe"
        "C:\ProgramData\miniconda3\Scripts\conda.exe"
    ) do (
        if exist %%~C (
            set "CONDA_EXE=%%~C"
            goto :conda_found
        )
    )
    echo [ERROR] Conda is not installed or not on PATH.
    echo Install Miniconda: https://docs.conda.io/en/latest/miniconda.html
    pause
    exit /b 1
)
:conda_found

REM Derive common paths
for %%D in ("%CONDA_EXE%") do set "CONDA_ROOT=%%~dp.."
set "ENV_NAME=terraagent"
set "ENV_DIR=%USERPROFILE%\.conda\envs\%ENV_NAME%"
set "ENV_PY=%ENV_DIR%\\python.exe"

REM Create environment if missing
if not exist "%ENV_PY%" (
    echo [INFO] Creating conda environment '%ENV_NAME%' from environment.yml ...
    "%CONDA_EXE%" env create -f environment.yml
    if %ERRORLEVEL% NEQ 0 (
        echo [ERROR] Environment creation failed.
        pause
        exit /b 1
    )
)

REM Run Streamlit using env python (avoids activation PATH issues)
if not exist "%ENV_PY%" (
    echo [ERROR] Expected python not found at %ENV_PY%
    pause
    exit /b 1
)

echo [INFO] Starting TerraAgent (streamlit) ...
echo     URL: http://localhost:8501
echo.
"%ENV_PY%" -m streamlit run streamlit_app.py --server.port 8501 --server.headless true --browser.gatherUsageStats false

REM Keep window open if something goes wrong
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] TerraAgent exited with a non-zero status.
    pause
)
