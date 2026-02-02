#!/bin/bash
# TerraAgent - Flood Analysis Launcher

echo "============================================"
echo "  TerraAgent - Flood Analysis Tool"
echo "============================================"

if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

echo "Starting Flood Analysis Interface..."
streamlit run generated_flood_app.py
