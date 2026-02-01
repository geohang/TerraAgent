#!/usr/bin/env bash
# ============================================================
# TerraAgent - One-click launcher (macOS/Linux)
# Double-click or run to create/activate the conda env and open the app
# ============================================================

set -e

echo ""
echo "========================================"
echo "  TerraAgent - Streamlit Platform"
echo "========================================"
echo ""

# Move to repo root (directory of this script)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Ensure conda exists
if ! command -v conda &> /dev/null; then
  echo "[ERROR] Conda is not installed or not on PATH."
  echo "Install Miniconda: https://docs.conda.io/en/latest/miniconda.html"
  read -p "Press Enter to exit..."
  exit 1
fi

# Enable conda in this shell
eval "$(conda shell.bash hook)"

# Create environment if missing
if ! conda env list | grep -q "^terraagent"; then
  echo "[INFO] Creating conda environment 'terraagent' from environment.yml ..."
  conda env create -f environment.yml
fi

echo "[INFO] Activating environment..."
conda activate terraagent

echo "[INFO] Starting TerraAgent (streamlit) ..."
python run.py

if [ $? -ne 0 ]; then
  echo ""
  echo "[ERROR] TerraAgent exited with a non-zero status."
  read -p "Press Enter to exit..."
fi
