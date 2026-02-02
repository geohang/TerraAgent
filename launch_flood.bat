@echo off
REM TerraAgent - Flood Analysis Launcher
REM Double-click this file to start the application

echo ============================================
echo   TerraAgent - Flood Analysis Tool
echo ============================================
echo.

if exist ".venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call .venv\Scripts\activate.bat
) else if exist "venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
)

echo Starting Flood Analysis Interface...
echo Press Ctrl+C to stop the server.
echo.

streamlit run generated_flood_app.py
pause
