"""
TerraAgent v0.1 - Earth Science Code-to-App Platform

This package contains the core modules for TerraAgent:
- science_climate: Climate change simulation
- science_fire: Wildfire risk assessment
- science_flood: Flood loss uncertainty analysis
- agent_builder: UI code generation engine
- main_platform: Unified Streamlit platform
- utils: Helper functions including Claude Code setup
"""

__version__ = "0.1.0"
__author__ = "TerraAgent Team"

# Expose key utilities for easy access
from .utils import (
    setup_claude_code,
    check_claude_code_installed,
    install_claude_code,
)

__all__ = [
    "setup_claude_code",
    "check_claude_code_installed", 
    "install_claude_code",
]
