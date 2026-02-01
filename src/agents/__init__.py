"""
TerraAgent v3.0 Agents

This module contains the three collaborative sub-agents:
- Inspector Agent: Analyzes code repositories
- Engineer Agent: Sets up environments and verifies code execution
- Designer Agent: Generates Streamlit UIs
"""

from .inspector import InspectorAgent
from .engineer import EngineerAgent
from .designer import DesignerAgent

__all__ = ["InspectorAgent", "EngineerAgent", "DesignerAgent"]
