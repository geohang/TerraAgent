"""
TerraAgent v2 - GitHub-to-App Platform

Phase 2: Remote GitHub fetch + instruction-aware UI generation.

Run with: streamlit run streamlit_app.py
"""

from __future__ import annotations

import os
import re
import sys
from typing import Optional

import streamlit as st

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from agent_builder import StreamlitBuilder  # noqa: E402
from utils import fetch_github_code, validate_api_key, get_reference_links  # noqa: E402

# Page configuration
st.set_page_config(
    page_title="TerraAgent Platform v2",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)


def get_llm_client(api_key: str, model_name: str):
    """Initialize an LLM client when a valid API key is provided."""
    if not validate_api_key(api_key):
        return None
    try:
        if "gpt" in model_name.lower():
            import openai

            return openai.OpenAI(api_key=api_key)
        if "claude" in model_name.lower():
            import anthropic

            return anthropic.Anthropic(api_key=api_key)
    except Exception:
        return None
    return None


def strip_set_page_config(code: str) -> str:
    """Remove the first st.set_page_config call to avoid duplicates during exec."""
    pattern = r"st\\.set_page_config\\(.*?\\)\\s*"
    return re.sub(pattern, "", code, count=1, flags=re.DOTALL)


def init_state():
    defaults = {
        "url_input": "",
        "instruction_input": "",
        "generated_code": "",
        "source_code": "",
        "last_error": "",
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


def main():
    init_state()

    st.title("🌍 TerraAgent Platform v2")
    st.caption("GitHub-to-App • Instruction-aware Streamlit generator")

    # === SIDEBAR CONFIG ===
    with st.sidebar:
        st.header("Configuration")
        api_key = st.text_input("API Key", type="password", placeholder="sk-...")
        model = st.selectbox("Model", ["gpt-4o", "claude-3.5-sonnet"], index=0)
        st.caption("If no key is provided, TerraAgent uses the built-in rule-based generator.")

        st.divider()
        st.subheader("Quick Start Demos")
        demos = {
            "🌤️ Climate": "src/science_climate.py",
            "🔥 Fire": "src/science_fire.py",
            "🌊 Flood": "src/science_flood.py",
        }
        for label, path in demos.items():
            if st.button(label, use_container_width=True):
                st.session_state.url_input = path
                st.session_state.instruction_input = "Use sidebar sliders for inputs and keep titles concise."
                st.info(f"Loaded demo: {label}")
                st.experimental_rerun()

        st.divider()
        st.subheader("Reference Projects")
        for ref in get_reference_links():
            st.markdown(f"- [{ref['name']}]({ref['url']}) — {ref['description']}")

    col1, col2 = st.columns([1.05, 1])

    # === LEFT: Inputs ===
    with col1:
        st.subheader("Input")
        url = st.text_input(
            "GitHub File URL (or local path)",
            key="url_input",
            placeholder="https://github.com/user/repo/blob/main/script.py",
        )
        instruction = st.text_area(
            "Natural Language Instruction",
            key="instruction_input",
            height=140,
            placeholder='e.g., "Use a slider for year, make the map red, add a Run button on the sidebar."',
        )

        if st.button("Generate App", type="primary", use_container_width=True):
            with st.spinner("Fetching code and generating UI..."):
                try:
                    code_str = fetch_github_code(url)
                    llm_client = get_llm_client(api_key, model)
                    builder = StreamlitBuilder(llm_client)
                    generated = builder.generate_ui_code(code_str, instruction)
                    st.session_state.generated_code = generated
                    st.session_state.source_code = code_str
                    st.session_state.last_error = ""
                    st.success("✅ App generated successfully!")
                except Exception as exc:  # noqa: BLE001
                    st.session_state.generated_code = ""
                    st.session_state.source_code = ""
                    st.session_state.last_error = str(exc)
                    st.error(f"Generation failed: {exc}")

    # === RIGHT: Preview & Code ===
    with col2:
        st.subheader("Preview & Code")
        tab_preview, tab_code = st.tabs(["Preview", "Code"])

        if st.session_state.generated_code:
            safe_code = strip_set_page_config(st.session_state.generated_code)

            with tab_preview:
                preview = st.container()
                try:
                    exec_globals = {"st": st, "__name__": "__exec__", "__file__": "generated_app.py"}
                    with preview:
                        exec(safe_code, exec_globals)
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Preview error: {exc}")

            with tab_code:
                st.code(st.session_state.generated_code, language="python")
                st.download_button(
                    "Download generated_app.py",
                    data=st.session_state.generated_code,
                    file_name="generated_app.py",
                    mime="text/x-python",
                    use_container_width=True,
                )
        else:
            with tab_preview:
                st.info(
                    "Paste a GitHub URL and instruction on the left, then click Generate to see the live app here."
                )
                if st.session_state.last_error:
                    st.error(f"Last error: {st.session_state.last_error}")
            with tab_code:
                st.code(
                    "# Generated app code will appear here after you click Generate.",
                    language="python",
                )


if __name__ == "__main__":
    main()
