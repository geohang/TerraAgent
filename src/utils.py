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
