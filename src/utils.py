"""
Utility functions for TerraAgent v2.

- fetch_github_code: download Python code from GitHub (blob/ -> raw) or a local file path.
- validate_api_key: lightweight sanity check for LLM API keys.
"""

from __future__ import annotations

import os
import re
import urllib.parse
from typing import Optional

import requests


def fetch_github_code(url: str, timeout: int = 15) -> str:
    """
    Fetch Python code from a GitHub URL or local path.

    Supports converting github.com/blob/... links to raw.githubusercontent.com.
    Falls back to reading a local file when the URL is a filesystem path.

    Args:
        url: GitHub file URL (blob or raw) or local file path.
        timeout: Request timeout in seconds.

    Returns:
        The fetched source code as a string.

    Raises:
        ValueError: If the URL/path cannot be resolved or fetched.
    """
    if not url or not url.strip():
        raise ValueError("Empty URL provided.")

    url = url.strip()

    # Local file support
    if "://" not in url and os.path.exists(url):
        with open(url, "r", encoding="utf-8") as f:
            return f.read()

    parsed = urllib.parse.urlparse(url)
    if not parsed.scheme:
        raise ValueError("Invalid URL. Provide a full GitHub URL or local path.")

    # Convert github.com blob links to raw
    raw_url = url
    if "github.com" in parsed.netloc and "/blob/" in parsed.path:
        raw_url = url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/", 1)

    try:
        resp = requests.get(raw_url, timeout=timeout)
        resp.raise_for_status()
        return resp.text
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Failed to fetch code from URL: {raw_url}") from exc


def validate_api_key(key: Optional[str]) -> bool:
    """
    Light validation for API keys (non-empty and plausible length/pattern).

    Args:
        key: The API key string.

    Returns:
        True if the key looks valid; False otherwise.
    """
    if not key:
        return False
    key = key.strip()
    if len(key) < 20:
        return False
    # Basic pattern for OpenAI/Anthropic style keys (starts with sk- or similar alnum)
    return bool(re.match(r"^[A-Za-z0-9\\-_]{20,}$", key))


def get_reference_links():
    """
    Return canonical reference project links defined in Target.md.

    Returns:
        list[dict]: Each dict has 'name', 'url', and 'description'.
    """
    return [
        {
            "name": "ClimateBench",
            "url": "https://github.com/duncanwp/ClimateBench",
            "description": "Climate model benchmarking suite (temperature anomaly heatmaps).",
        },
        {
            "name": "Fire Weather Index (FWI)",
            "url": "https://github.com/steidani/FireWeatherIndex",
            "description": "Wildfire risk calculation and mapping.",
        },
        {
            "name": "UNSAFE Framework",
            "url": "https://github.com/abpoll/unsafe",
            "description": "Flood loss uncertainty and probabilistic risk assessment.",
        },
    ]


def check_claude_code_installed() -> dict:
    """
    Check if Claude Code CLI is installed and available.
    
    Returns:
        dict with 'installed', 'version', and 'install_instructions'
    """
    import subprocess
    import platform
    
    try:
        result = subprocess.run(
            ["claude", "--version"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            version = result.stdout.strip()
            return {
                "installed": True,
                "version": version,
                "install_instructions": None
            }
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    
    # Determine install instructions based on OS
    system = platform.system()
    if system == "Windows":
        install_cmd = 'irm https://claude.ai/install.ps1 | iex'
    else:
        install_cmd = 'curl -fsSL https://claude.ai/install.sh | bash'
    
    return {
        "installed": False,
        "version": None,
        "install_instructions": install_cmd
    }


def install_claude_code() -> bool:
    """
    Install Claude Code CLI.
    
    Opens the installation page or runs the install command.
    
    Returns:
        True if installation was initiated, False otherwise.
    """
    import subprocess
    import platform
    import webbrowser
    
    status = check_claude_code_installed()
    
    if status["installed"]:
        print(f"✅ Claude Code is already installed: {status['version']}")
        return True
    
    system = platform.system()
    
    print("📦 Installing Claude Code CLI...")
    print(f"   Platform: {system}")
    
    try:
        if system == "Windows":
            # Open PowerShell and run install command
            print("   Running: irm https://claude.ai/install.ps1 | iex")
            subprocess.Popen(
                ["powershell", "-Command", "irm https://claude.ai/install.ps1 | iex"],
                creationflags=subprocess.CREATE_NEW_CONSOLE
            )
            print("✅ Installation started in new PowerShell window")
            print("   After installation, restart your terminal and run 'claude' to authenticate")
        else:
            # macOS/Linux
            print("   Running: curl -fsSL https://claude.ai/install.sh | bash")
            subprocess.run(
                ["bash", "-c", "curl -fsSL https://claude.ai/install.sh | bash"],
                check=True
            )
            print("✅ Installation complete!")
            print("   Run 'claude' to authenticate with your Anthropic account")
        
        return True
        
    except Exception as e:
        print(f"❌ Automatic installation failed: {e}")
        print("\n📖 Manual Installation:")
        print(f"   {status['install_instructions']}")
        print("\n   Or visit: https://claude.ai/code")
        webbrowser.open("https://claude.ai/code")
        return False


def setup_claude_code():
    """
    Interactive setup for Claude Code.
    
    Checks installation, installs if needed, and provides next steps.
    """
    print("=" * 60)
    print("  Claude Code Setup for TerraAgent")
    print("=" * 60)
    print()
    
    status = check_claude_code_installed()
    
    if status["installed"]:
        print(f"✅ Claude Code is installed: {status['version']}")
        print()
        print("🚀 Next Steps:")
        print("   1. Open TerraAgent in VS Code: code .")
        print("   2. Ask Claude Code to integrate a package:")
        print('      "Integrate UNSAFE from github.com/abpoll/unsafe"')
        print()
        return True
    
    print("❌ Claude Code is not installed")
    print()
    
    response = input("Would you like to install Claude Code now? [y/N]: ").strip().lower()
    
    if response in ('y', 'yes'):
        return install_claude_code()
    else:
        print()
        print("📖 To install later:")
        print(f"   {status['install_instructions']}")
        print()
        print("💡 You can still use TerraAgent without Claude Code!")
        print("   Use the web interface with LLM APIs (OpenAI, Anthropic, etc.)")
        print("   Run: streamlit run streamlit_app.py")
        return False
