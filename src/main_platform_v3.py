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
    page_title="TerraAgent v1.0",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Custom CSS for terminal-like log display
TERMINAL_CSS = """
<style>
.terminal-container {
    background-color: #101413;
    border-radius: 12px;
    padding: 14px;
    font-family: 'JetBrains Mono', 'SFMono-Regular', 'Consolas', monospace;
    font-size: 12px;
    line-height: 1.55;
    max-height: 320px;
    overflow-y: auto;
    border: 1px solid rgba(255, 255, 255, 0.08);
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.25);
}
.terminal-line {
    margin: 2px 0;
    white-space: pre-wrap;
    word-wrap: break-word;
}
.log-info { color: #7cb7ff; }
.log-success { color: #8be6b1; }
.log-error { color: #ff8e8e; }
.log-warning { color: #ffd479; }
.log-header { color: #d7b4ff; font-weight: 600; }
</style>
"""

# Global UI styling
GLOBAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:wght@600;700&family=Space+Grotesk:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg-1: #f9f6f0;
    --bg-2: #eef4f1;
    --ink: #1f2a24;
    --muted: #5a6b60;
    --accent: #2f6b5f;
    --accent-2: #e3a73d;
    --card: rgba(255, 255, 255, 0.86);
    --border: rgba(31, 42, 36, 0.12);
    --shadow: 0 12px 30px rgba(31, 42, 36, 0.08);
    --radius: 14px;
}

html, body, [data-testid="stAppViewContainer"] {
    font-family: 'Space Grotesk', sans-serif;
    color: var(--ink);
    background:
        radial-gradient(1100px 600px at 10% -10%, #e3f2ec 0%, transparent 60%),
        radial-gradient(900px 600px at 90% 0%, #f7e6cf 0%, transparent 55%),
        linear-gradient(180deg, var(--bg-1) 0%, var(--bg-2) 100%);
}

[data-testid="stSidebar"] {
    background: #f4f7f5;
    border-right: 1px solid var(--border);
}

h1, h2, h3, h4, h5 {
    font-family: 'Fraunces', serif;
    letter-spacing: -0.01em;
}

.block-container {
    padding-top: 1.8rem;
    max-width: 1200px;
}

.hero {
    background: var(--card);
    border: 1px solid var(--border);
    box-shadow: var(--shadow);
    border-radius: var(--radius);
    padding: 1.6rem 1.8rem;
    margin-bottom: 1.25rem;
    animation: fadeInUp 0.6s ease-out;
}
.hero-eyebrow {
    font-size: 0.8rem;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: var(--muted);
}
.hero-title {
    font-size: 3.1rem;
    font-weight: 700;
    margin: 0.3rem 0 0.2rem;
}
.hero-sub {
    font-size: 1.05rem;
    color: var(--muted);
}
.hero-cta {
    margin-top: 0.7rem;
    font-size: 0.95rem;
    color: var(--ink);
}

.step-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 0.85rem 0.95rem;
    box-shadow: var(--shadow);
}
.step-title {
    font-weight: 600;
    margin-bottom: 0.25rem;
}
.step-meta {
    font-size: 0.9rem;
    color: var(--muted);
}

.status-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-left: 4px solid var(--accent);
    border-radius: 12px;
    padding: 0.65rem 0.8rem;
    box-shadow: var(--shadow);
    margin-bottom: 0.6rem;
}
.status-card.warning { border-left-color: #b86a2a; }
.status-card.info { border-left-color: #2a6f8a; }
.status-title {
    font-weight: 600;
    font-size: 0.95rem;
}
.status-detail {
    font-size: 0.85rem;
    color: var(--muted);
}

.stButton > button {
    background: var(--accent);
    color: #fff;
    border-radius: 999px;
    border: none;
    padding: 0.45rem 1.2rem;
    font-weight: 600;
    transition: transform 0.15s ease, box-shadow 0.15s ease, background 0.15s ease;
    box-shadow: 0 8px 20px rgba(47, 107, 95, 0.25);
}
.stButton > button:hover {
    background: #245b50;
    transform: translateY(-1px);
}

div[data-baseweb="tab-list"] {
    gap: 0.5rem;
}
div[data-baseweb="tab-list"] button {
    background: var(--card);
    border-radius: 999px;
    border: 1px solid var(--border);
    padding: 0.3rem 0.9rem;
}
div[data-baseweb="tab-list"] button[aria-selected="true"] {
    background: var(--accent);
    color: #fff;
}

input, textarea, select {
    border-radius: 10px !important;
}

[data-testid="stMetric"] {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 0.5rem 0.75rem;
    box-shadow: var(--shadow);
}

@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

@media (max-width: 768px) {
    .hero-title { font-size: 2.2rem; }
    .block-container { padding-top: 1.2rem; }
}
</style>
"""


# Supported LLM providers - users can type any model version
LLM_PROVIDERS = {
    "Claude Code (CLI)": {
        "default_model": "sonnet",
        "placeholder": "e.g., sonnet, opus, haiku",
        "key_prefix": "",
        "key_hint": "Uses Claude Code CLI (must be installed)",
        "docs_url": "https://code.claude.com/docs/en/overview",
    },
    "OpenAI": {
        "default_model": "gpt-4o",
        "placeholder": "e.g., gpt-4o, gpt-4-turbo, o1-preview",
        "key_prefix": "sk-",
        "key_hint": "sk-...",
        "docs_url": "https://platform.openai.com/docs/models",
    },
    "Anthropic": {
        "default_model": "claude-sonnet-4-20250514",
        "placeholder": "e.g., claude-sonnet-4-20250514, claude-opus-4-20250514",
        "key_prefix": "sk-ant-",
        "key_hint": "sk-ant-...",
        "docs_url": "https://docs.anthropic.com/en/docs/about-claude/models",
    },
    "Google": {
        "default_model": "gemini-2.0-flash",
        "placeholder": "e.g., gemini-2.0-flash, gemini-1.5-pro",
        "key_prefix": "AI",
        "key_hint": "AIza...",
        "docs_url": "https://ai.google.dev/models/gemini",
    },
    "Mistral": {
        "default_model": "mistral-large-latest",
        "placeholder": "e.g., mistral-large-latest, codestral-latest",
        "key_prefix": "",
        "key_hint": "API key",
        "docs_url": "https://docs.mistral.ai/getting-started/models/",
    },
    "OpenRouter": {
        "default_model": "openai/gpt-4o",
        "placeholder": "e.g., openai/gpt-4o, anthropic/claude-3.5-sonnet",
        "key_prefix": "sk-or-",
        "key_hint": "sk-or-...",
        "docs_url": "https://openrouter.ai/models",
    },
    "Ollama (Local)": {
        "default_model": "llama3.2",
        "placeholder": "e.g., llama3.2, codellama, mistral, deepseek-coder",
        "key_prefix": "",
        "key_hint": "No key needed",
        "docs_url": "https://ollama.com/library",
    },
}


def get_llm_client(api_key: str, provider: str, model_name: str) -> Optional[Any]:
    """Initialize an LLM client based on provider and model."""
    try:
        if provider == "Claude Code (CLI)":
            # Claude Code uses CLI, return a special marker
            return {"client": None, "model": model_name, "provider": "claude_code"}
        
        elif provider == "OpenAI":
            import openai
            return {"client": openai.OpenAI(api_key=api_key), "model": model_name, "provider": "openai"}

        elif provider == "Anthropic":
            import anthropic
            return {"client": anthropic.Anthropic(api_key=api_key), "model": model_name, "provider": "anthropic"}

        elif provider == "Google":
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            return {"client": genai, "model": model_name, "provider": "google"}

        elif provider == "Mistral":
            from mistralai import Mistral
            return {"client": Mistral(api_key=api_key), "model": model_name, "provider": "mistral"}

        elif provider == "OpenRouter":
            import openai
            client = openai.OpenAI(
                api_key=api_key,
                base_url="https://openrouter.ai/api/v1"
            )
            return {"client": client, "model": model_name, "provider": "openrouter"}

        elif provider == "Ollama (Local)":
            import openai
            client = openai.OpenAI(
                api_key="ollama",
                base_url="http://localhost:11434/v1"
            )
            return {"client": client, "model": model_name, "provider": "ollama"}

    except Exception as e:
        st.warning(f"Failed to initialize {provider} client: {e}")
        return None
    return None


def check_claude_code_installed() -> bool:
    """Check if Claude Code CLI is installed and accessible."""
    import subprocess
    try:
        result = subprocess.run(
            ["claude", "--version"],
            capture_output=True,
            text=True,
            timeout=10
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def call_claude_code(prompt: str, model: str = "sonnet", working_dir: str = None) -> str:
    """
    Call Claude Code CLI to generate code.
    
    Args:
        prompt: The prompt to send to Claude Code
        model: Model alias (sonnet, opus, haiku)
        working_dir: Working directory for the command
    
    Returns:
        Generated code as string
    """
    import subprocess
    import json
    
    cmd = [
        "claude",
        "-p",  # Print mode (non-interactive)
        "--model", model,
        "--output-format", "text",
        "--dangerously-skip-permissions",  # Skip permission prompts for automation
        prompt
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,  # 2 minute timeout
            cwd=working_dir
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"Claude Code error: {result.stderr}")
        
        return result.stdout.strip()
        
    except subprocess.TimeoutExpired:
        raise RuntimeError("Claude Code timed out after 120 seconds")
    except FileNotFoundError:
        raise RuntimeError("Claude Code CLI not found. Install it from: https://code.claude.com/docs/en/setup")


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


def render_status_card(status: Dict[str, str]) -> str:
    """Render a small status card (HTML)."""
    tone = status.get("tone", "info")
    title = status.get("title", "")
    detail = status.get("detail", "")
    return (
        f'<div class="status-card {tone}">'
        f'<div class="status-title">{title}</div>'
        f'<div class="status-detail">{detail}</div>'
        f'</div>'
    )


def resolve_llm_client(
    provider: str,
    model: str,
    api_key: str
) -> tuple[Optional[Any], Dict[str, str]]:
    """Return initialized LLM client (if ready) and a status payload."""
    if provider == "Claude Code (CLI)":
        if check_claude_code_installed():
            client = get_llm_client("", provider, model)
            return client, {
                "tone": "success",
                "title": "Claude Code ready",
                "detail": f"Model: {model}"
            }
        return None, {
            "tone": "warning",
            "title": "Claude Code not installed",
            "detail": "Install the CLI or switch provider. Using rule-based generation."
        }

    if provider == "Ollama (Local)":
        client = get_llm_client("ollama", provider, model)
        return client, {
            "tone": "info" if client else "warning",
            "title": "Local LLM mode",
            "detail": "Ensure Ollama is running on localhost:11434."
        }

    if not api_key:
        return None, {
            "tone": "info",
            "title": "Rule-based generation",
            "detail": "Add an API key to enable LLM-assisted UI generation."
        }

    if not validate_api_key(api_key):
        return None, {
            "tone": "warning",
            "title": "API key looks invalid",
            "detail": "Using rule-based generation until a valid key is provided."
        }

    client = get_llm_client(api_key, provider, model)
    if client:
        return client, {
            "tone": "success",
            "title": "LLM ready",
            "detail": f"{provider} · {model}"
        }

    return None, {
        "tone": "warning",
        "title": "LLM init failed",
        "detail": "Falling back to rule-based generation."
    }


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
    st.session_state.phase = "verifying"
    add_log("=" * 50)
    add_log("PHASE 2: ENVIRONMENT VERIFICATION")
    add_log("=" * 50)
    
    engineer = EngineerAgent(log_callback=add_log)
    result = None
    
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
        result = None
    
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
        add_log(f"  Return type: {analysis.get('return_type', 'unknown')}")
        add_log(f"  Parameters: {[p['name'] for p in analysis.get('parameters', [])]}")
        
        # Log which generation method is being used
        if llm_client:
            provider = llm_client.get('provider', 'unknown') if isinstance(llm_client, dict) else 'unknown'
            model = llm_client.get('model', 'unknown') if isinstance(llm_client, dict) else 'unknown'
            add_log(f"🤖 Using LLM: {provider} / {model}")
            add_log("Generating Streamlit UI with AI assistance...")
        else:
            add_log("📋 Using rule-based generation (no API key provided)")
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
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

    st.markdown(
        """
        <div class="hero">
            <div class="hero-eyebrow">Agentic Earth Science Platform</div>
            <div class="hero-title">TerraAgent v1.0 • Agentic Edition</div>
            <div class="hero-sub">
                Transform scientific Python packages into production-ready Streamlit apps
                with maps, metrics, and uncertainty quantification.
            </div>
            <div class="hero-cta">
                Paste a GitHub repo, describe the app you want, and generate a working UI in minutes.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    step_cols = st.columns(3)
    with step_cols[0]:
        st.markdown(
            """
            <div class="step-card">
                <div class="step-title">1. Inspect</div>
                <div class="step-meta">Clone and analyze scientific codebases</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with step_cols[1]:
        st.markdown(
            """
            <div class="step-card">
                <div class="step-title">2. Engineer</div>
                <div class="step-meta">Install deps and verify key functions</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with step_cols[2]:
        st.markdown(
            """
            <div class="step-card">
                <div class="step-title">3. Design</div>
                <div class="step-meta">Generate a rich Streamlit interface</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    # === SIDEBAR ===
    with st.sidebar:
        st.markdown("### Control Center")
        st.caption("Connect a provider or run rule-based generation.")

        # Provider selection
        provider = st.selectbox(
            "LLM Provider",
            list(LLM_PROVIDERS.keys()),
            index=0,
            help="Select your LLM provider"
        )

        # Model input - user can type any model version
        provider_config = LLM_PROVIDERS[provider]
        model = st.text_input(
            "Model Name",
            value=provider_config["default_model"],
            placeholder=provider_config["placeholder"],
            help=f"Enter any {provider} model name. [View available models]({provider_config['docs_url']})"
        )

        # API key input (disable when not needed)
        key_hint = LLM_PROVIDERS[provider]["key_hint"]
        key_disabled = provider in {"Ollama (Local)", "Claude Code (CLI)"}
        api_key = st.text_input(
            "API Key",
            type="password",
            placeholder=key_hint,
            value="" if key_disabled else st.session_state.get("api_key", ""),
            disabled=key_disabled,
            help="Not required for local/CLI providers" if key_disabled else f"Enter your {provider} API key"
        )

        llm_client, llm_status = resolve_llm_client(provider, model, api_key)
        st.markdown(render_status_card(llm_status), unsafe_allow_html=True)
        st.caption("No key? Rule-based generation still works.")

        st.divider()

        with st.expander("🌐 GitHub Examples", expanded=True):
            st.caption("Clone real repositories to generate apps")
            github_demos = {
                "🌤️ ClimateBench": {
                    "url": "https://github.com/duncanwp/ClimateBench",
                    "instruction": "Analyze the climate emulator code and create an interactive app for climate prediction visualization."
                },
                "🔥 FireWeatherIndex": {
                    "url": "https://github.com/steidani/FireWeatherIndex",
                    "instruction": "Create a fire weather index calculator with inputs for temperature, humidity, wind speed, and precipitation."
                },
                "🌊 UNSAFE Flood": {
                    "url": "https://github.com/abpoll/unsafe",
                    "instruction": "Build a flood risk assessment tool with uncertainty quantification and Monte Carlo visualization."
                }
            }

            for label, demo in github_demos.items():
                if st.button(label, use_container_width=True, key=f"github_{label}"):
                    st.session_state.github_url = demo["url"]
                    st.session_state.instruction = demo["instruction"]
                    st.rerun()

        with st.expander("📁 Local Demos", expanded=False):
            st.caption("Built-in examples (no download needed)")
            local_demos = {
                "🌤️ Climate": {
                    "path": "src/science_climate.py",
                    "instruction": "Create a climate projection app with location dropdown, target year slider (2024-2100), emission scenario selector, and display warming results on a map."
                },
                "🔥 Fire Risk": {
                    "path": "src/science_fire.py",
                    "instruction": "Create a fire risk assessment app with location dropdown, temperature, humidity, wind speed, precipitation inputs, and display FWI results on a map with risk level."
                },
                "🌊 Flood Loss": {
                    "path": "src/science_flood.py",
                    "instruction": "Create a flood loss calculator with location dropdown, flood depth slider, and display mean loss, confidence intervals, and location on a map."
                }
            }

            for label, demo in local_demos.items():
                if st.button(label, use_container_width=True, key=f"local_{label}"):
                    st.session_state.github_url = demo["path"]
                    st.session_state.instruction = demo["instruction"]
                    run_demo_mode(demo["path"], demo["instruction"], llm_client)
                    st.rerun()

        with st.expander("📚 Reference Projects", expanded=False):
            for ref in get_reference_links():
                st.markdown(f"- [{ref['name']}]({ref['url']})")
    
    # === MAIN CONTENT ===

    st.subheader("Start a Build")
    st.caption("Paste a GitHub repository URL or local module path, then describe the app you want.")

    col_input1, col_input2 = st.columns([1, 1])

    with col_input1:
        github_url = st.text_input(
            "Repository Source",
            value=st.session_state.github_url,
            placeholder="https://github.com/user/repo or src/science_flood.py",
            help="Enter a GitHub repository URL or a local module path"
        )
        st.session_state.github_url = github_url

    with col_input2:
        instruction = st.text_area(
            "App Instruction",
            value=st.session_state.instruction,
            height=120,
            placeholder="e.g., 'Build a flood loss calculator with a map preview and uncertainty histogram'",
            help="Describe how you want the generated app to look and behave"
        )
        st.session_state.instruction = instruction

    # Generate Button
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])

    with col_btn1:
        if st.button("🚀 Generate App", type="primary", use_container_width=True):
            if github_url:
                is_url = github_url.startswith("http")
                is_local = os.path.exists(github_url)
                if not is_url and not is_local:
                    st.error("Please enter a valid GitHub URL or a local file path.")
                else:
                    if is_url:
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
            progress_map = {
                "idle": 0,
                "cloning": 20,
                "installing": 35,
                "verifying": 50,
                "building": 80,
                "done": 100
            }
            st.progress(progress_map.get(phase, 0))
            status_map = {
                "idle": ("⚪", "Idle"),
                "cloning": ("🔄", "Cloning Repository..."),
                "installing": ("🔄", "Setting up Environment..."),
                "verifying": ("🔄", "Verifying & Testing..."),
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
