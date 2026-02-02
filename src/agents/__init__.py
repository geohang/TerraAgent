"""
TerraAgent v3.0 Agents

This module contains the collaborative sub-agents:
- Inspector Agent: Analyzes code repositories
- Engineer Agent: Sets up environments and verifies code execution
- Designer Agent: Generates Streamlit UIs
- Integrator Agent: Orchestrates full package integration workflow

Workflow:
    GitHub URL → Install → Analyze → Generate Wrapper → Create Interface
"""

from .inspector import InspectorAgent, RepoSummary, FunctionInfo, ClassInfo
from .engineer import EngineerAgent, VerificationResult
from .designer import DesignerAgent, GeneratedApp
from .integrator import IntegratorAgent, IntegrationResult, PackageAnalysis

__all__ = [
    # Agents
    "InspectorAgent",
    "EngineerAgent", 
    "DesignerAgent",
    "IntegratorAgent",
    # Data classes
    "RepoSummary",
    "FunctionInfo",
    "ClassInfo",
    "VerificationResult",
    "GeneratedApp",
    "IntegrationResult",
    "PackageAnalysis",
]
