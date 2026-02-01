"""
Streamlit entrypoint for TerraAgent.

Run with: `streamlit run streamlit_app.py`
This delegates to the unified platform in src/main_platform.py.
"""

from src.main_platform import main


if __name__ == "__main__":
    main()
