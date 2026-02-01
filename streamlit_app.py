"""
TerraAgent v3.0: Intelligent Earth Science SaaS Platform

Run with: `streamlit run streamlit_app.py`

This delegates to the v3.0 agentic platform with:
- Inspector Agent: Clones and analyzes repositories
- Engineer Agent: Sets up environments and verifies code
- Designer Agent: Generates Streamlit UIs

For the legacy v2.0 platform, import from src.main_platform instead.
"""

import os
import sys

# Ensure src is in path
src_path = os.path.join(os.path.dirname(__file__), "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Use v3.0 by default
from src.main_platform_v3 import main


if __name__ == "__main__":
    main()
