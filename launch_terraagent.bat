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
    REM Fallback to local venv if conda is unavailable
    set "ENV_PY="
    if exist ".venv\Scripts\python.exe" set "ENV_PY=.venv\Scripts\python.exe"
    if "%ENV_PY%"=="" if exist "venv\Scripts\python.exe" set "ENV_PY=venv\Scripts\python.exe"
    if not "%ENV_PY%"=="" goto :env_found

    echo [ERROR] Conda is not installed or not on PATH, and no local venv was found.
    echo Install Miniconda: https://docs.conda.io/en/latest/miniconda.html
    echo Or create a venv: python -m venv .venv ^&^& .venv\Scripts\python -m pip install -r requirements.txt
    pause
    exit /b 1
)
:conda_found

REM Derive common paths
for %%D in ("%CONDA_EXE%") do set "CONDA_ROOT=%%~dpD.."
set "ENV_NAME=terraagent"

REM Check multiple possible environment locations
set "ENV_PY="
for %%E in (
    "%USERPROFILE%\anaconda3\envs\%ENV_NAME%\python.exe"
    "%USERPROFILE%\miniconda3\envs\%ENV_NAME%\python.exe"
    "%USERPROFILE%\.conda\envs\%ENV_NAME%\python.exe"
    "C:\ProgramData\Anaconda3\envs\%ENV_NAME%\python.exe"
    "C:\ProgramData\miniconda3\envs\%ENV_NAME%\python.exe"
) do (
    if exist %%~E (
        set "ENV_PY=%%~E"
        goto :env_found
    )
)

REM Environment not found, create it
echo [INFO] Creating conda environment '%ENV_NAME%' from environment.yml ...
"%CONDA_EXE%" env create -f environment.yml -n %ENV_NAME%
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Environment creation failed.
    pause
    exit /b 1
)

REM Re-check for environment after creation
for %%E in (
    "%USERPROFILE%\anaconda3\envs\%ENV_NAME%\python.exe"
    "%USERPROFILE%\miniconda3\envs\%ENV_NAME%\python.exe"
    "%USERPROFILE%\.conda\envs\%ENV_NAME%\python.exe"
    "C:\ProgramData\Anaconda3\envs\%ENV_NAME%\python.exe"
    "C:\ProgramData\miniconda3\envs\%ENV_NAME%\python.exe"
) do (
    if exist %%~E (
        set "ENV_PY=%%~E"
        goto :env_found
    )
)

echo [ERROR] Could not find python.exe in the created environment.
pause
exit /b 1

:env_found
echo [INFO] Using environment: %ENV_PY%

REM Run Streamlit using env python (avoids activation PATH issues)
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
