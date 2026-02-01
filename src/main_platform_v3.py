"""
TerraAgent v3.0: Intelligent Earth Science SaaS Platform (Agentic Edition)

This is the unified platform that orchestrates the three collaborative agents:
- Inspector Agent: Clones repos and analyzes code structure
- Engineer Agent: Sets up environments and verifies code execution
- Designer Agent: Generates Streamlit UIs from verified code

Features:
- GitHub URL input with automatic cloning
- VS Code-like terminal output for real-time logs
- Live app preview with exec()
- Demo mode with built-in science modules

Run with: streamlit run streamlit_app.py
"""

from __future__ import annotations

import os
import re
import sys
import uuid
import time
from typing import Any, Dict, List, Optional
from dataclasses import asdict

import streamlit as st

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import agents
from agents import InspectorAgent, EngineerAgent, DesignerAgent
from agent_builder import StreamlitBuilder
from utils import validate_api_key, get_reference_links


# Page configuration
st.set_page_config(
    page_title="TerraAgent v3.0",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Custom CSS for terminal-like log display
TERMINAL_CSS = """
<style>
.terminal-container {
    background-color: #1e1e1e;
    border-radius: 8px;
    padding: 12px;
    font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
    font-size: 12px;
    line-height: 1.5;
    max-height: 300px;
    overflow-y: auto;
    border: 1px solid #333;
}
.terminal-line {
    margin: 2px 0;
    white-space: pre-wrap;
    word-wrap: break-word;
}
.log-info { color: #3794ff; }
.log-success { color: #89d185; }
.log-error { color: #f14c4c; }
.log-warning { color: #cca700; }
.log-header { color: #c586c0; font-weight: bold; }
</style>
"""


def get_llm_client(api_key: str, model_name: str) -> Optional[Any]:
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


def format_log_message(msg: str) -> str:
    """Format log message with color coding."""
    if msg.startswith("="):
        return f'<div class="terminal-line log-header">{msg}</div>'
    elif "✓" in msg or "SUCCESS" in msg:
        return f'<div class="terminal-line log-success">{msg}</div>'
    elif "✗" in msg or "ERROR" in msg or "failed" in msg.lower():
        return f'<div class="terminal-line log-error">{msg}</div>'
    elif "Warning" in msg or "⚠" in msg:
        return f'<div class="terminal-line log-warning">{msg}</div>'
    else:
        return f'<div class="terminal-line log-info">{msg}</div>'


def render_terminal(logs: List[str]) -> str:
    """Render logs as a VS Code-like terminal."""
    html_logs = "".join([format_log_message(log) for log in logs])
    return f'{TERMINAL_CSS}<div class="terminal-container">{html_logs}</div>'


def strip_set_page_config(code: str) -> str:
    """Remove st.set_page_config call to avoid duplicates during exec."""
    pattern = r"st\.set_page_config\([^)]*\)\s*"
    return re.sub(pattern, "", code, count=1, flags=re.DOTALL)


def init_state():
    """Initialize session state with defaults."""
    defaults = {
        "github_url": "",
        "instruction": "",
        "generated_code": "",
        "logs": [],
        "phase": "idle",  # idle, cloning, installing, verifying, building, done
        "session_id": str(uuid.uuid4())[:8],
        "repo_summary": None,
        "verification_result": None,
        "last_error": "",
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


def add_log(message: str):
    """Add a message to the log."""
    timestamp = time.strftime("%H:%M:%S")
    st.session_state.logs.append(f"[{timestamp}] {message}")


def clear_logs():
    """Clear the log messages."""
    st.session_state.logs = []


def run_agentic_pipeline(github_url: str, instruction: str, llm_client: Optional[Any]):
    """
    Run the full agentic pipeline:
    Phase 1: Code Ingestion (Inspector)
    Phase 2: Environment Verification (Engineer)
    Phase 3: UI Generation (Designer)
    """
    clear_logs()
    session_id = st.session_state.session_id
    
    # Phase 1: Code Ingestion
    st.session_state.phase = "cloning"
    add_log("=" * 50)
    add_log("PHASE 1: CODE INGESTION")
    add_log("=" * 50)
    
    inspector = InspectorAgent(llm_client=llm_client, log_callback=add_log)
    
    try:
        summary = inspector.inspect(github_url, session_id)
        st.session_state.repo_summary = summary
        add_log(f"✓ Repository analyzed: {len(summary.python_files)} Python files found")
    except Exception as e:
        add_log(f"✗ Inspection failed: {e}")
        st.session_state.last_error = str(e)
        st.session_state.phase = "idle"
        return
    
    # Select a function to verify
    target_function = None
    if summary.functions:
        # Prefer functions that return plt.Figure
        for func in summary.functions:
            if func.return_type and "Figure" in func.return_type:
                target_function = asdict(func)
                break
        if not target_function:
            target_function = asdict(summary.functions[0])
        add_log(f"Selected target function: {target_function['name']}")
    
    # Phase 2: Environment Verification
    st.session_state.phase = "installing"
    add_log("=" * 50)
    add_log("PHASE 2: ENVIRONMENT VERIFICATION")
    add_log("=" * 50)
    
    engineer = EngineerAgent(log_callback=add_log)
    
    try:
        result = engineer.verify(summary.repo_path, target_function)
        st.session_state.verification_result = result
        
        if result.success:
            add_log("✓ SMOKE TEST PASSED!")
        else:
            add_log(f"⚠ Verification had issues: {result.error_log[:200] if result.error_log else 'Unknown'}")
    except Exception as e:
        add_log(f"✗ Verification failed: {e}")
        st.session_state.verification_result = None
    
    # Phase 3: UI Generation
    st.session_state.phase = "building"
    add_log("=" * 50)
    add_log("PHASE 3: UI GENERATION")
    add_log("=" * 50)
    
    designer = DesignerAgent(llm_client=llm_client, log_callback=add_log)
    
    try:
        if target_function:
            working_snippet = result.working_snippet if result and result.success else None
            app = designer.design(
                target_function,
                summary.repo_path,
                instruction,
                working_snippet
            )
            st.session_state.generated_code = app.code
            add_log(f"✓ Generated {len(app.code)} characters of Streamlit code")
        else:
            add_log("⚠ No suitable function found for UI generation")
            
    except Exception as e:
        add_log(f"✗ UI generation failed: {e}")
        st.session_state.last_error = str(e)
    
    st.session_state.phase = "done"
    add_log("=" * 50)
    add_log("PIPELINE COMPLETE")
    add_log("=" * 50)


def run_demo_mode(demo_path: str, instruction: str, llm_client: Optional[Any]):
    """
    Run in demo mode using built-in science modules.
    """
    clear_logs()
    add_log("=" * 50)
    add_log("DEMO MODE: Local Science Module")
    add_log("=" * 50)
    
    try:
        # Read the demo file
        full_path = os.path.join(os.path.dirname(__file__), "..", demo_path)
        if not os.path.exists(full_path):
            full_path = demo_path
            
        add_log(f"Loading demo from: {demo_path}")
        
        with open(full_path, 'r') as f:
            code_str = f.read()
        
        add_log(f"✓ Loaded {len(code_str)} characters of code")
        
        # Use the existing StreamlitBuilder for demos
        builder = StreamlitBuilder(llm_client)
        
        add_log("Analyzing code structure...")
        analysis = builder.analyze_code(code_str)
        add_log(f"✓ Found function: {analysis.get('name', 'unknown')}")
        
        add_log("Generating Streamlit UI...")
        generated = builder.generate_ui_code(code_str, instruction)
        
        st.session_state.generated_code = generated
        add_log(f"✓ Generated {len(generated)} characters of Streamlit code")
        
        st.session_state.phase = "done"
        add_log("=" * 50)
        add_log("DEMO COMPLETE")
        add_log("=" * 50)
        
    except Exception as e:
        add_log(f"✗ Demo failed: {e}")
        st.session_state.last_error = str(e)
        st.session_state.phase = "idle"


def main():
    """Main application entry point."""
    init_state()
    
    # Title
    st.title("🌍 TerraAgent v3.0")
    st.caption("Intelligent Earth Science SaaS Platform • Agentic Edition")
    
    # === SIDEBAR ===
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        api_key = st.text_input(
            "LLM API Key",
            type="password",
            placeholder="sk-... or sk-ant-...",
            help="OpenAI or Anthropic API key for enhanced generation"
        )
        
        model = st.selectbox(
            "Model",
            ["gpt-4o", "claude-3.5-sonnet"],
            index=0
        )
        
        st.caption("Without an API key, TerraAgent uses rule-based generation.")
        
        st.divider()
        
        # Quick Start Demos
        st.subheader("🚀 Quick Start Demos")
        
        demos = {
            "🌤️ Climate": {
                "path": "src/science_climate.py",
                "instruction": "Create a climate simulation app with year slider and emission scenario selector."
            },
            "🔥 Fire Risk": {
                "path": "src/science_fire.py", 
                "instruction": "Create a fire risk assessment app with temperature, humidity, wind, and rain inputs."
            },
            "🌊 Flood Loss": {
                "path": "src/science_flood.py",
                "instruction": "Create a flood loss calculator with Monte Carlo simulation visualization."
            }
        }
        
        for label, demo in demos.items():
            if st.button(label, use_container_width=True, key=f"demo_{label}"):
                st.session_state.github_url = demo["path"]
                st.session_state.instruction = demo["instruction"]
                llm_client = get_llm_client(api_key, model)
                run_demo_mode(demo["path"], demo["instruction"], llm_client)
                st.rerun()
        
        st.divider()
        
        # Reference Links
        st.subheader("📚 Reference Projects")
        for ref in get_reference_links():
            st.markdown(f"- [{ref['name']}]({ref['url']})")
    
    # === MAIN CONTENT ===
    
    # Input Section
    st.subheader("📥 Input")
    
    col_input1, col_input2 = st.columns([1, 1])
    
    with col_input1:
        github_url = st.text_input(
            "GitHub Repository URL",
            value=st.session_state.github_url,
            placeholder="https://github.com/user/repo",
            help="Enter a GitHub repository URL to clone and analyze"
        )
        st.session_state.github_url = github_url
    
    with col_input2:
        instruction = st.text_area(
            "Natural Language Instruction",
            value=st.session_state.instruction,
            height=100,
            placeholder="e.g., 'Build a fire risk warning system with a red heatmap and year slider in the sidebar'",
            help="Describe how you want the generated app to look and behave"
        )
        st.session_state.instruction = instruction
    
    # Generate Button
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])
    
    with col_btn1:
        if st.button("🚀 Generate App", type="primary", use_container_width=True):
            if github_url:
                llm_client = get_llm_client(api_key, model)
                
                # Check if it's a local demo path or a GitHub URL
                if github_url.startswith("http"):
                    run_agentic_pipeline(github_url, instruction, llm_client)
                else:
                    run_demo_mode(github_url, instruction, llm_client)
                st.rerun()
            else:
                st.error("Please enter a GitHub URL or select a demo")
    
    with col_btn2:
        if st.button("🗑️ Clear", use_container_width=True):
            for key in ["generated_code", "logs", "repo_summary", "verification_result", "last_error"]:
                if key == "logs":
                    st.session_state[key] = []
                else:
                    st.session_state[key] = "" if key in ["generated_code", "last_error"] else None
            st.session_state.phase = "idle"
            st.rerun()
    
    st.divider()
    
    # Terminal Output & Preview
    col_terminal, col_preview = st.columns([1, 1.2])
    
    with col_terminal:
        st.subheader("🖥️ Agent Terminal")
        
        if st.session_state.logs:
            # Status indicator
            phase = st.session_state.phase
            status_map = {
                "idle": ("⚪", "Idle"),
                "cloning": ("🔄", "Cloning Repository..."),
                "installing": ("🔄", "Installing Dependencies..."),
                "verifying": ("🔄", "Running Smoke Test..."),
                "building": ("🔄", "Generating UI..."),
                "done": ("✅", "Complete")
            }
            icon, text = status_map.get(phase, ("⚪", "Unknown"))
            st.markdown(f"**Status:** {icon} {text}")
            
            # Render terminal
            st.markdown(render_terminal(st.session_state.logs), unsafe_allow_html=True)
        else:
            st.info("Agent logs will appear here when you generate an app.")
    
    with col_preview:
        st.subheader("📱 App Preview")
        
        tab_preview, tab_code = st.tabs(["Live Preview", "Generated Code"])
        
        with tab_preview:
            if st.session_state.generated_code:
                try:
                    safe_code = strip_set_page_config(st.session_state.generated_code)
                    
                    # Create execution context
                    exec_globals = {
                        "st": st,
                        "__name__": "__exec__",
                        "__file__": "generated_app.py"
                    }
                    
                    # Execute in a container
                    preview_container = st.container()
                    with preview_container:
                        exec(safe_code, exec_globals)
                        
                except Exception as e:
                    st.error(f"Preview error: {e}")
                    st.code(str(e), language="text")
            else:
                st.info("Generated app will appear here after you click Generate.")
                
                if st.session_state.last_error:
                    st.error(f"Last error: {st.session_state.last_error}")
        
        with tab_code:
            if st.session_state.generated_code:
                st.code(st.session_state.generated_code, language="python")
                
                st.download_button(
                    "📥 Download generated_app.py",
                    data=st.session_state.generated_code,
                    file_name="generated_app.py",
                    mime="text/x-python",
                    use_container_width=True
                )
            else:
                st.code("# Generated code will appear here...", language="python")


if __name__ == "__main__":
    main()
