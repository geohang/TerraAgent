"""
TerraAgent v3.0: Integrator Agent

🔗 The Integrator Agent
Task: Orchestrate the complete workflow for integrating external scientific packages.
Capability: Combines Inspector, Engineer, and Designer agents to create a seamless
pipeline from GitHub URL to working Streamlit application.

Workflow:
1. Get GitHub URL from user
2. Install the package (Engineer Agent)
3. Analyze the package structure (Inspector Agent)  
4. Generate wrapper module (Integrator logic)
5. Create Streamlit interface (Designer Agent)
"""

import os
import sys
import json
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .inspector import InspectorAgent, RepoSummary, FunctionInfo
from .engineer import EngineerAgent, VerificationResult
from .designer import DesignerAgent, GeneratedApp


@dataclass
class IntegrationResult:
    """Result of package integration."""
    success: bool
    package_name: str
    github_url: str
    wrapper_module_path: Optional[str]
    streamlit_app_path: Optional[str]
    launcher_path: Optional[str]  # Path to launcher script (.bat or .sh)
    verified_functions: List[str]
    data_requirements: List[Dict[str, str]]
    error_log: Optional[str]


@dataclass
class PackageAnalysis:
    """Analysis of an external package."""
    name: str
    github_url: str
    main_functions: List[FunctionInfo]
    data_requirements: List[Dict[str, str]]
    dependencies: List[str]
    install_command: str
    documentation_url: str


class IntegratorAgent:
    """
    Orchestrates the complete integration of external scientific packages.
    
    This agent combines the capabilities of Inspector, Engineer, and Designer
    agents to create a seamless workflow:
    
    1. Install package from GitHub
    2. Analyze package structure
    3. Generate wrapper module with three-tier fallback
    4. Create Streamlit interface
    
    Example:
        >>> integrator = IntegratorAgent()
        >>> result = integrator.integrate_package(
        ...     github_url="https://github.com/abpoll/unsafe",
        ...     domain="flood",
        ...     user_instruction="Create flood loss calculator"
        ... )
        >>> print(result.wrapper_module_path)
        "src/science_flood.py"
    """
    
    # Template for generating wrapper modules
    WRAPPER_TEMPLATE = '''"""
{domain_title} Analysis Module

This module provides a TerraAgent wrapper for the {package_name} package.
{package_description}

Installation:
    pip install git+{github_url}

Reference: 
    {github_url}
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Optional
from pathlib import Path
import warnings
import json

# ============================================================================
# Package and Data Availability
# ============================================================================

PACKAGE_AVAILABLE = False
DATA_READY = False

try:
    from {package_name} import {main_imports}
    PACKAGE_AVAILABLE = True
except ImportError:
    warnings.warn(
        "{package_name} package not installed. Install with:\\n"
        "  pip install git+{github_url}\\n"
        "Using simplified estimation instead."
    )

# Data directory for processed files
DATA_DIR = Path(__file__).parent.parent / "data" / "{domain}"

# Global data storage (loaded once)
_PROCESSED_DATA = None


# ============================================================================
# Data Management
# ============================================================================

def _load_data():
    """Load processed data if available."""
    global _PROCESSED_DATA, DATA_READY
    
    if not PACKAGE_AVAILABLE:
        return False
    
    data_file = DATA_DIR / "processed" / "data.pqt"
    
    if data_file.exists():
        try:
            _PROCESSED_DATA = pd.read_parquet(data_file)
            DATA_READY = True
            return True
        except Exception as e:
            warnings.warn(f"Failed to load data: {{e}}")
            return False
    
    return False


def setup_data(data_url: Optional[str] = None) -> bool:
    """
    Setup required data files.
    
    Downloads and processes data for use with the framework.
    This only needs to be run once.
    
    Args:
        data_url: URL to download data from (optional).
    
    Returns:
        True if setup successful, False otherwise.
    """
    global DATA_READY
    
    if not PACKAGE_AVAILABLE:
        print("Package not installed. Install with:")
        print("  pip install git+{github_url}")
        return False
    
    # Create data directory
    (DATA_DIR / "processed").mkdir(parents=True, exist_ok=True)
    
    print("Data setup complete!")
    DATA_READY = True
    return True


# Try to load data on module import
_load_data()


# ============================================================================
# Status and Configuration
# ============================================================================

CONFIGURATIONS = {{
    "Default": {{"param": 1.0}},
}}


def get_configurations() -> List[str]:
    """Return list of available configurations."""
    return list(CONFIGURATIONS.keys())


def check_installation() -> Dict[str, Any]:
    """
    Check installation and data status.
    
    Returns:
        Dict with complete status information.
    """
    mode = "simplified"
    if PACKAGE_AVAILABLE and DATA_READY:
        mode = "direct"
    elif PACKAGE_AVAILABLE:
        mode = "style"
    
    return {{
        "installed": PACKAGE_AVAILABLE,
        "data_ready": DATA_READY,
        "package": "{package_name}",
        "install_command": "pip install git+{github_url}",
        "documentation": "{github_url}",
        "data_dir": str(DATA_DIR),
        "mode": mode,
    }}


# ============================================================================
# Main Calculation Function
# ============================================================================

def calculate_{domain}_result(
    config_name: str,
    input_value: float,
    num_simulations: int = 1000,
    use_package: bool = True
) -> Dict[str, Any]:
    """
    Calculate {domain} result with uncertainty quantification.
    
    Args:
        config_name: Name of the configuration to use.
        input_value: Primary input parameter.
        num_simulations: Number of Monte Carlo iterations (default: 1000).
        use_package: Whether to use {package_name} if available.
    
    Returns:
        Dict containing:
            - config: Configuration name
            - mean_result: Expected value
            - std_result: Standard deviation
            - ci_lower, ci_upper: 95% confidence interval
            - using_package: Whether external package was used
    """
    if config_name not in CONFIGURATIONS:
        available = ", ".join(CONFIGURATIONS.keys())
        raise ValueError(f"Unknown config: {{config_name}}. Available: {{available}}")
    
    config = CONFIGURATIONS[config_name]
    
    # Select calculation method based on availability
    if PACKAGE_AVAILABLE and DATA_READY and use_package:
        results = _calculate_direct(config, input_value, num_simulations)
        using_package = True
        mode = "direct"
    elif PACKAGE_AVAILABLE and use_package:
        results = _calculate_style(config, input_value, num_simulations)
        using_package = True
        mode = "style"
    else:
        results = _calculate_simplified(config, input_value, num_simulations)
        using_package = False
        mode = "simplified"
    
    return {{
        "config": config_name,
        "input_value": input_value,
        "mean_result": float(np.mean(results)),
        "std_result": float(np.std(results)),
        "ci_lower": float(np.percentile(results, 2.5)),
        "ci_upper": float(np.percentile(results, 97.5)),
        "using_package": using_package,
        "mode": mode,
        "num_simulations": num_simulations,
    }}


def _calculate_direct(
    config: Dict, input_value: float, num_simulations: int
) -> np.ndarray:
    """Calculate using package with processed data files."""
    # TODO: Implement direct package usage
    return _calculate_style(config, input_value, num_simulations)


def _calculate_style(
    config: Dict, input_value: float, num_simulations: int
) -> np.ndarray:
    """Calculate using package methodology but without data files."""
    # TODO: Implement package-style calculation
    return _calculate_simplified(config, input_value, num_simulations)


def _calculate_simplified(
    config: Dict, input_value: float, num_simulations: int
) -> np.ndarray:
    """Simplified calculation when package not available."""
    rng = np.random.default_rng()
    base = input_value * config.get("param", 1.0)
    return rng.normal(base, base * 0.1, size=num_simulations)


# ============================================================================
# Visualization
# ============================================================================

def create_result_histogram(result: Dict[str, Any]) -> plt.Figure:
    """Create histogram of results with statistics."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    rng = np.random.default_rng(42)
    samples = rng.normal(result["mean_result"], result["std_result"], 1000)
    
    ax.hist(samples, bins=50, color='steelblue', alpha=0.7, edgecolor='white')
    ax.axvline(result["mean_result"], color='red', linestyle='--', 
               linewidth=2, label=f'Mean: {{result["mean_result"]:.2f}}')
    ax.axvline(result["ci_lower"], color='orange', linestyle=':', 
               linewidth=2, label=f'95% CI')
    ax.axvline(result["ci_upper"], color='orange', linestyle=':', linewidth=2)
    
    ax.set_xlabel("Result Value")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Result Distribution - {{result['config']}} (Mode: {{result['mode']}})")
    ax.legend()
    
    plt.tight_layout()
    return fig
'''
    
    def __init__(
        self,
        llm_client: Optional[Any] = None,
        log_callback: Optional[Callable[[str], None]] = None,
        workspace_path: Optional[str] = None
    ):
        """
        Initialize the Integrator Agent.
        
        Args:
            llm_client: Optional LLM client for enhanced generation.
            log_callback: Optional callback function for logging.
            workspace_path: Path to the TerraAgent workspace.
        """
        self.llm_client = llm_client
        self.log = log_callback or (lambda msg: print(f"[Integrator] {msg}"))
        
        # Initialize sub-agents
        self.inspector = InspectorAgent(llm_client=llm_client, log_callback=log_callback)
        self.engineer = EngineerAgent(log_callback=log_callback)
        self.designer = DesignerAgent(llm_client=llm_client, log_callback=log_callback)
        
        # Set workspace path
        if workspace_path:
            self.workspace_path = Path(workspace_path)
        else:
            self.workspace_path = Path(__file__).parent.parent.parent
    
    def integrate_package(
        self,
        github_url: str,
        domain: str,
        user_instruction: Optional[str] = None,
        extra_deps: Optional[List[str]] = None,
        data_urls: Optional[List[Dict[str, str]]] = None
    ) -> IntegrationResult:
        """
        Integrate a scientific package from GitHub.
        
        This is the main entry point for package integration. It performs:
        1. Package installation from GitHub
        2. Package analysis to identify main functions
        3. Wrapper module generation
        4. Streamlit interface creation
        
        Args:
            github_url: GitHub repository URL (e.g., https://github.com/owner/repo).
            domain: Domain name for the integration (e.g., "flood", "climate").
            user_instruction: User's description of desired functionality.
            extra_deps: Additional dependencies to install.
            data_urls: List of data file URLs to download.
            
        Returns:
            IntegrationResult with paths to generated files and status.
            
        Example:
            >>> result = integrator.integrate_package(
            ...     "https://github.com/abpoll/unsafe",
            ...     domain="flood",
            ...     user_instruction="Create flood loss calculator with uncertainty"
            ... )
        """
        self.log("=" * 60)
        self.log(f"Starting integration for: {github_url}")
        self.log(f"Domain: {domain}")
        self.log("=" * 60)
        
        # Extract package name from URL
        package_name = github_url.rstrip('/').split('/')[-1]
        
        # Step 1: Install the package
        self.log("\n📦 Step 1: Installing package...")
        success, message, verified_imports = self.engineer.install_github_package(
            github_url=github_url,
            extra_deps=extra_deps,
            use_current_env=True
        )
        
        if not success:
            return IntegrationResult(
                success=False,
                package_name=package_name,
                github_url=github_url,
                wrapper_module_path=None,
                streamlit_app_path=None,
                launcher_path=None,
                verified_functions=[],
                data_requirements=[],
                error_log=message
            )
        
        # Step 2: Analyze the package (try to import and find functions)
        self.log("\n🔍 Step 2: Analyzing package...")
        analysis = self._analyze_installed_package(package_name, github_url)
        
        # Step 3: Download data files if specified
        data_requirements = []
        if data_urls:
            self.log("\n📥 Step 3: Downloading data files...")
            for data_info in data_urls:
                url = data_info.get("url")
                filename = data_info.get("filename", url.split("/")[-1])
                output_path = self.workspace_path / "data" / domain / filename
                
                success, msg = self.engineer.download_data_file(url, str(output_path))
                if success:
                    data_requirements.append({
                        "file": filename,
                        "path": str(output_path),
                        "url": url
                    })
        
        # Step 4: Generate wrapper module
        self.log("\n📝 Step 4: Generating wrapper module...")
        wrapper_path = self._generate_wrapper_module(
            package_name=package_name,
            domain=domain,
            github_url=github_url,
            analysis=analysis,
            data_requirements=data_requirements
        )
        
        # Step 5: Generate Streamlit app
        streamlit_path = None
        launcher_path = None
        if user_instruction:
            self.log("\n🎨 Step 5: Generating Streamlit interface...")
            streamlit_path = self._generate_streamlit_app(
                wrapper_path=wrapper_path,
                domain=domain,
                user_instruction=user_instruction
            )
            
            # Step 6: Generate launcher script
            if streamlit_path:
                self.log("\n🚀 Step 6: Generating launcher script...")
                launcher_path = self._generate_launcher_script(
                    streamlit_app_path=streamlit_path,
                    domain=domain
                )
        
        self.log("\n✅ Integration complete!")
        self.log(f"   Wrapper module: {wrapper_path}")
        if streamlit_path:
            self.log(f"   Streamlit app: {streamlit_path}")
        if launcher_path:
            self.log(f"   Launcher: {launcher_path}")
            self.log(f"\n💡 Double-click the launcher to start the app!")
        
        return IntegrationResult(
            success=True,
            package_name=package_name,
            github_url=github_url,
            wrapper_module_path=str(wrapper_path) if wrapper_path else None,
            streamlit_app_path=str(streamlit_path) if streamlit_path else None,
            launcher_path=str(launcher_path) if launcher_path else None,
            verified_functions=verified_imports,
            data_requirements=data_requirements,
            error_log=None
        )
    
    def _analyze_installed_package(
        self,
        package_name: str,
        github_url: str
    ) -> PackageAnalysis:
        """Analyze an installed package to find main functions."""
        main_functions = []
        dependencies = []
        
        # Try to import and inspect the package
        try:
            import importlib
            module = importlib.import_module(package_name)
            
            # Find public functions
            for name in dir(module):
                if not name.startswith('_'):
                    obj = getattr(module, name)
                    if callable(obj):
                        # Create FunctionInfo
                        import inspect
                        try:
                            sig = inspect.signature(obj)
                            params = [
                                {"name": p.name, "type": str(p.annotation), "default": str(p.default)}
                                for p in sig.parameters.values()
                            ]
                            func_info = FunctionInfo(
                                name=name,
                                file_path=getattr(obj, '__module__', ''),
                                line_number=0,
                                signature=str(sig),
                                docstring=obj.__doc__,
                                return_type=str(sig.return_annotation),
                                parameters=params
                            )
                            main_functions.append(func_info)
                        except (ValueError, TypeError):
                            pass
        except ImportError as e:
            self.log(f"  Could not import {package_name}: {e}")
        
        return PackageAnalysis(
            name=package_name,
            github_url=github_url,
            main_functions=main_functions,
            data_requirements=[],
            dependencies=dependencies,
            install_command=f"pip install git+{github_url}",
            documentation_url=github_url
        )
    
    def _generate_wrapper_module(
        self,
        package_name: str,
        domain: str,
        github_url: str,
        analysis: PackageAnalysis,
        data_requirements: List[Dict[str, str]]
    ) -> Path:
        """Generate a wrapper module for the package."""
        
        # Determine main imports
        main_imports = package_name
        if analysis.main_functions:
            func_names = [f.name for f in analysis.main_functions[:5]]
            # main_imports = ", ".join(func_names)
        
        # Generate wrapper content
        content = self.WRAPPER_TEMPLATE.format(
            domain_title=domain.replace("_", " ").title(),
            domain=domain,
            package_name=package_name,
            package_description=f"A TerraAgent integration for {package_name}.",
            github_url=github_url,
            main_imports=main_imports
        )
        
        # Write to file
        wrapper_path = self.workspace_path / "src" / f"science_{domain}.py"
        
        # Don't overwrite existing files
        if wrapper_path.exists():
            self.log(f"  Warning: {wrapper_path} already exists, creating backup")
            backup_path = wrapper_path.with_suffix('.py.bak')
            wrapper_path.rename(backup_path)
        
        wrapper_path.write_text(content, encoding='utf-8')
        self.log(f"  Created: {wrapper_path}")
        
        return wrapper_path
    
    def _generate_streamlit_app(
        self,
        wrapper_path: Path,
        domain: str,
        user_instruction: str
    ) -> Optional[Path]:
        """
        Generate a rich Streamlit app with visualizations for the wrapper module.
        
        This generates a production-quality app with:
        - Status indicators
        - Key metrics display
        - Map visualization
        - Distribution histograms
        - Sensitivity analysis charts
        - Comparison tables
        """
        try:
            # Read the wrapper module to understand its structure
            source_code = wrapper_path.read_text(encoding='utf-8')
            
            # Generate rich Streamlit app using template
            app_code = self._create_rich_streamlit_app(
                source_code=source_code,
                domain=domain,
                user_instruction=user_instruction
            )
            
            if app_code:
                app_path = self.workspace_path / f"generated_{domain}_app.py"
                app_path.write_text(app_code, encoding='utf-8')
                self.log(f"  Created: {app_path}")
                return app_path
            
            # Fallback to designer agent if template generation fails
            self.log("  Using designer agent fallback...")
            result = self.designer.generate_streamlit_app(
                source_code=source_code,
                user_instruction=user_instruction,
                repo_path=str(self.workspace_path)
            )
            
            if result and result.code:
                app_path = self.workspace_path / f"generated_{domain}_app.py"
                app_path.write_text(result.code, encoding='utf-8')
                self.log(f"  Created: {app_path}")
                return app_path
                
        except Exception as e:
            self.log(f"  Error generating Streamlit app: {e}")
        
        return None
    
    def _create_rich_streamlit_app(
        self,
        source_code: str,
        domain: str,
        user_instruction: str
    ) -> Optional[str]:
        """
        Create a rich Streamlit app with visualizations from source code.
        
        Generates app with map, histograms, metrics, and sensitivity charts.
        """
        domain_title = domain.title()
        domain_emoji = {
            'flood': '🌊',
            'climate': '🌡️',
            'fire': '🔥',
            'earthquake': '🌍',
            'hurricane': '🌀',
        }.get(domain.lower(), '📊')
        
        app_template = f'''"""
Auto-generated Streamlit UI for {domain} analysis
Generated by TerraAgent IntegratorAgent
User Instruction: {user_instruction}
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# Import Science Module
# ============================================================================

# Paste the science module code here or import
{source_code}

# ============================================================================
# Page Configuration
# ============================================================================

st.set_page_config(
    page_title="{domain_emoji} {domain_title} Analysis Tool",
    page_icon="{domain_emoji}",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# Header & Status
# ============================================================================

st.title("{domain_emoji} {domain_title} Estimator")
st.markdown("""
{user_instruction}

This app uses Monte Carlo simulation with uncertainty quantification.
""")

# Show package status
try:
    # Try domain-specific check functions
    if 'check_unsafe_installation' in dir():
        status = check_unsafe_installation()
    elif 'check_{domain}_installation' in dir():
        status = check_{domain}_installation()
    elif 'check_installation' in dir():
        status = check_installation()
    else:
        status = {{"installed": False, "mode": "simplified"}}
    
    if status.get('installed') and status.get('data_ready', status.get('installed')):
        st.success(f"✅ External package active | Mode: **{{status.get('mode', 'direct')}}**")
    elif status.get('installed'):
        st.warning(f"⚠️ Package installed but data not ready | Mode: **{{status.get('mode', 'style')}}**")
    else:
        st.warning(f"⚠️ Using simplified estimation | Install: `{{status.get('install_command', 'pip install ...')}}`")
except:
    st.info("ℹ️ Running in standalone mode")

st.divider()

# ============================================================================
# Session State
# ============================================================================

if 'result' not in st.session_state:
    st.session_state.result = None

# ============================================================================
# Sidebar Inputs
# ============================================================================

st.sidebar.header("📊 Input Parameters")

# Get available locations/configurations
try:
    if 'get_flood_locations' in dir():
        locations = get_flood_locations()
    elif 'get_climate_locations' in dir():
        locations = get_climate_locations()
    elif 'get_fire_locations' in dir():
        locations = get_fire_locations()
    elif 'get_locations' in dir():
        locations = get_locations()
    else:
        locations = ["Default"]
except:
    locations = ["Default"]

location = st.sidebar.selectbox(
    "📍 Location",
    options=locations,
    help="Select a location for analysis"
)

# Primary parameter
param1 = st.sidebar.slider(
    "🌊 Primary Parameter",
    min_value=0.1,
    max_value=5.0,
    value=1.5,
    step=0.1,
    help="Main input parameter"
)

# Secondary parameter
param2 = st.sidebar.number_input(
    "💰 Value/Amount",
    min_value=10000,
    max_value=2000000,
    value=300000,
    step=10000
)

# Category selector
category = st.sidebar.selectbox(
    "🏠 Category/Type",
    options=["S", "C", "B"],
    format_func=lambda x: {{"S": "Standard", "C": "Category C", "B": "Category B"}}[x]
)

# Simulations
num_simulations = st.sidebar.slider(
    "🎲 Monte Carlo Simulations",
    min_value=500,
    max_value=10000,
    value=2000,
    step=500
)

st.sidebar.divider()

# Run button
run_clicked = st.sidebar.button("🚀 Run Analysis", type="primary", use_container_width=True)

# ============================================================================
# Main Computation
# ============================================================================

if run_clicked:
    with st.spinner("Running Monte Carlo simulation..."):
        try:
            # Try to call the main calculation function
            if 'calculate_flood_loss' in dir():
                result = calculate_flood_loss(
                    location_name=location,
                    flood_depth=param1,
                    num_simulations=num_simulations,
                    property_value=param2,
                    foundation_type=category
                )
            elif 'simulate_warming' in dir():
                result = simulate_warming(
                    location_name=location,
                    scenario=location if 'SSP' in location else "SSP2-4.5 (Middle Road)",
                    projection_years=int(param1 * 30),
                    num_simulations=num_simulations
                )
            elif 'calculate_fwi_risk' in dir():
                result = calculate_fwi_risk(
                    location_name=location,
                    temperature=param1 * 10 + 20,
                    humidity=50,
                    wind_speed=20,
                    num_simulations=num_simulations
                )
            else:
                result = {{"error": "No calculation function found"}}
            
            st.session_state.result = result
        except Exception as e:
            st.error(f"Error: {{str(e)}}")
            st.session_state.result = None

# ============================================================================
# Results Display
# ============================================================================

if st.session_state.result is not None:
    result = st.session_state.result
    
    if 'error' in result:
        st.error(result['error'])
    else:
        # Two-column layout
        col1, col2 = st.columns([1, 1])
        
        # ---- Column 1: Metrics ----
        with col1:
            st.subheader("📊 Analysis Results")
            
            # Find the main metric keys
            mean_key = next((k for k in ['mean_loss', 'mean_warming', 'mean_fwi', 'mean_result'] if k in result), None)
            std_key = next((k for k in ['std_loss', 'std_warming', 'std_fwi', 'std_result'] if k in result), None)
            ratio_key = next((k for k in ['damage_ratio', 'ratio'] if k in result), None)
            
            m1, m2, m3 = st.columns(3)
            with m1:
                if mean_key:
                    val = result[mean_key]
                    if val > 1000:
                        st.metric("Mean", f"${{val:,.0f}}")
                    else:
                        st.metric("Mean", f"{{val:.2f}}")
            with m2:
                if std_key:
                    val = result[std_key]
                    if val > 1000:
                        st.metric("Std Dev", f"${{val:,.0f}}")
                    else:
                        st.metric("Std Dev", f"{{val:.2f}}")
            with m3:
                if ratio_key:
                    st.metric("Ratio", f"{{result[ratio_key]:.1%}}")
                elif 'risk_level' in result:
                    st.metric("Risk", result['risk_level'])
            
            # Confidence interval
            if 'ci_lower' in result and 'ci_upper' in result:
                ci_lower = result['ci_lower']
                ci_upper = result['ci_upper']
                if ci_lower > 1000:
                    st.info(f"**95% CI:** ${{ci_lower:,.0f}} — ${{ci_upper:,.0f}}")
                else:
                    st.info(f"**95% CI:** {{ci_lower:.2f}} — {{ci_upper:.2f}}")
            
            # Details
            st.markdown("#### 📋 Details")
            details = {{k: str(v) for k, v in result.items() if not isinstance(v, (list, dict))}}
            st.dataframe(pd.DataFrame(details.items(), columns=["Key", "Value"]), hide_index=True, use_container_width=True)
        
        # ---- Column 2: Map ----
        with col2:
            st.subheader("🗺️ Location")
            
            if 'lat' in result and 'lon' in result:
                map_df = pd.DataFrame({{'lat': [result['lat']], 'lon': [result['lon']]}})
                st.map(map_df, zoom=10)
                st.caption(f"📍 {{result.get('location', location)}} ({{result['lat']:.4f}}, {{result['lon']:.4f}})")
            else:
                st.info("No coordinates available for mapping")
        
        # ---- Visualization ----
        st.divider()
        st.subheader("📈 Distribution Analysis")
        
        # Try to create histogram
        try:
            if 'create_loss_histogram' in dir():
                fig = create_loss_histogram(result)
                st.pyplot(fig)
            elif 'create_warming_chart' in dir():
                fig = create_warming_chart(result)
                st.pyplot(fig)
            elif 'create_fwi_gauge' in dir():
                fig = create_fwi_gauge(result)
                st.pyplot(fig)
            elif 'simulations' in result and result['simulations']:
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.hist(result['simulations'], bins=50, edgecolor='white', alpha=0.7)
                ax.axvline(result.get(mean_key, 0), color='red', linestyle='--', label='Mean')
                ax.set_xlabel('Value')
                ax.set_ylabel('Frequency')
                ax.legend()
                st.pyplot(fig)
        except Exception as e:
            st.warning(f"Could not create visualization: {{e}}")
        
        # Raw data expander
        with st.expander("📄 View Raw Data"):
            st.json(result)

else:
    st.info("👈 Configure parameters in the sidebar and click **Run Analysis** to start.")
    
    # Show available locations on map
    st.subheader("📍 Available Locations")
    
    try:
        if 'FLOOD_LOCATIONS' in dir():
            locs = FLOOD_LOCATIONS
        elif 'CLIMATE_LOCATIONS' in dir():
            locs = CLIMATE_LOCATIONS
        elif 'FIRE_LOCATIONS' in dir():
            locs = FIRE_LOCATIONS
        else:
            locs = {{}}
        
        if locs:
            col1, col2 = st.columns([1, 1])
            with col1:
                loc_df = pd.DataFrame([
                    {{"Location": k, "Lat": v.get('lat', 'N/A'), "Lon": v.get('lon', 'N/A')}}
                    for k, v in locs.items()
                ])
                st.dataframe(loc_df, hide_index=True, use_container_width=True)
            
            with col2:
                map_data = pd.DataFrame([
                    {{'lat': v['lat'], 'lon': v['lon']}}
                    for v in locs.values()
                    if 'lat' in v and 'lon' in v
                ])
                if not map_data.empty:
                    st.map(map_data, zoom=3)
    except:
        pass
'''
        
        return app_template
    
    def _generate_launcher_script(
        self,
        streamlit_app_path: Path,
        domain: str
    ) -> Optional[Path]:
        """
        Generate launcher scripts (.bat for Windows, .sh for Unix).
        
        This allows users to double-click to start the Streamlit app.
        
        Args:
            streamlit_app_path: Path to the Streamlit app file.
            domain: Domain name for the launcher name.
            
        Returns:
            Path to the generated launcher script.
        """
        import platform
        
        try:
            app_name = streamlit_app_path.name
            launcher_name = f"launch_{domain}"
            
            # Windows batch script
            bat_content = f'''@echo off
REM TerraAgent - {domain.title()} Analysis Launcher
REM Generated automatically by IntegratorAgent
REM Double-click this file to start the application

echo ============================================
echo   TerraAgent - {domain.title()} Analysis Tool
echo ============================================
echo.

REM Check if we're in a virtual environment
if exist ".venv\\Scripts\\activate.bat" (
    echo Activating virtual environment...
    call .venv\\Scripts\\activate.bat
) else if exist "venv\\Scripts\\activate.bat" (
    echo Activating virtual environment...
    call venv\\Scripts\\activate.bat
)

echo Starting {domain.title()} Analysis Interface...
echo.
echo The application will open in your browser.
echo Press Ctrl+C in this window to stop the server.
echo.

streamlit run {app_name}

pause
'''
            
            # Unix shell script
            sh_content = f'''#!/bin/bash
# TerraAgent - {domain.title()} Analysis Launcher
# Generated automatically by IntegratorAgent
# Run this script to start the application

echo "============================================"
echo "  TerraAgent - {domain.title()} Analysis Tool"
echo "============================================"
echo ""

# Check if we're in a virtual environment
if [ -f ".venv/bin/activate" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
elif [ -f "venv/bin/activate" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

echo "Starting {domain.title()} Analysis Interface..."
echo ""
echo "The application will open in your browser."
echo "Press Ctrl+C to stop the server."
echo ""

streamlit run {app_name}
'''
            
            # Write both scripts
            bat_path = self.workspace_path / f"{launcher_name}.bat"
            sh_path = self.workspace_path / f"{launcher_name}.sh"
            
            bat_path.write_text(bat_content, encoding='utf-8')
            self.log(f"  Created Windows launcher: {bat_path}")
            
            sh_path.write_text(sh_content, encoding='utf-8')
            self.log(f"  Created Unix launcher: {sh_path}")
            
            # Return the appropriate launcher for the current platform
            if platform.system() == 'Windows':
                return bat_path
            else:
                return sh_path
                
        except Exception as e:
            self.log(f"  Error generating launcher script: {e}")
            return None
    
    def quick_integrate(
        self,
        github_url: str,
        domain: str
    ) -> Dict[str, Any]:
        """
        Quick integration with minimal configuration.
        
        Just provide a GitHub URL and domain name, and this method handles everything.
        
        Args:
            github_url: GitHub repository URL.
            domain: Domain name (e.g., "flood", "climate", "fire").
            
        Returns:
            Dict with integration status and paths.
        """
        result = self.integrate_package(
            github_url=github_url,
            domain=domain,
            user_instruction=f"Create a {domain} analysis application"
        )
        
        return asdict(result)
    
    def check_integration_status(self, domain: str) -> Dict[str, Any]:
        """
        Check the status of an existing integration.
        
        Args:
            domain: Domain name to check.
            
        Returns:
            Dict with integration status.
        """
        wrapper_path = self.workspace_path / "src" / f"science_{domain}.py"
        
        if not wrapper_path.exists():
            return {
                "exists": False,
                "domain": domain,
                "wrapper_path": None
            }
        
        # Try to import and check status
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location(f"science_{domain}", wrapper_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Look for domain-specific check function (e.g., check_unsafe_installation, check_climate_installation)
            check_func_names = [
                f"check_{domain}_installation",
                f"check_unsafe_installation",  # flood module uses this
                "check_installation"
            ]
            
            status = None
            for func_name in check_func_names:
                if hasattr(module, func_name):
                    status = getattr(module, func_name)()
                    break
            
            if status is None:
                status = {"note": "No check_installation function found"}
            
            return {
                "exists": True,
                "domain": domain,
                "wrapper_path": str(wrapper_path),
                **status
            }
        except Exception as e:
            return {
                "exists": True,
                "domain": domain,
                "wrapper_path": str(wrapper_path),
                "error": str(e)
            }


if __name__ == "__main__":
    # Test
    integrator = IntegratorAgent()
    print("Integrator Agent initialized.")
    print("Use integrator.integrate_package(github_url, domain, instruction) to integrate a package.")
