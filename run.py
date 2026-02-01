"""
TerraAgent Launcher

Quick launch script for the TerraAgent platform.
Run this file to start the Streamlit application.
"""

import subprocess
import sys
import os


def main():
    """Launch the TerraAgent Streamlit application."""
    # Get the path to main_platform.py
    script_dir = os.path.dirname(os.path.abspath(__file__))
    main_platform_path = os.path.join(script_dir, "src", "main_platform.py")

    # Run streamlit
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        main_platform_path,
        "--server.headless", "true"
    ])


if __name__ == "__main__":
    main()
