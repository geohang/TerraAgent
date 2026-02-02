# TerraAgent Module Templates

This document provides templates for creating science modules that wrap external packages.

## Template 1: Basic Science Module

Use this template for simple packages with minimal data requirements.

```python
"""
{Domain} Analysis Module

This module provides a TerraAgent wrapper for the {PACKAGE_NAME} package.
{Brief description of what the package does}

Installation:
    pip install git+{github_url}

Reference: 
    {Paper citation or documentation URL}

Keywords: {Relevant keywords for discoverability}
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Optional
from pathlib import Path
import warnings

# ============================================================================
# Package Availability Check
# ============================================================================

PACKAGE_AVAILABLE = False

try:
    from {package} import {main_function}
    PACKAGE_AVAILABLE = True
except ImportError:
    warnings.warn(
        "{PACKAGE_NAME} package not installed. Install with:\n"
        "  pip install git+{github_url}\n"
        "Using simplified estimation instead."
    )


# ============================================================================
# Configuration
# ============================================================================

# Sample configurations for demo purposes
CONFIGURATIONS = {
    "Config 1": {"param_a": 1.0, "param_b": 100},
    "Config 2": {"param_a": 2.5, "param_b": 200},
    "Config 3": {"param_a": 5.0, "param_b": 500},
}


def get_configurations() -> List[str]:
    """Return list of available configurations."""
    return list(CONFIGURATIONS.keys())


def check_installation() -> Dict[str, Any]:
    """
    Check package installation status.
    
    Returns:
        Dict with installation status and configuration info.
    """
    return {
        "installed": PACKAGE_AVAILABLE,
        "package": "{package_name}",
        "install_command": "pip install git+{github_url}",
        "documentation": "{github_url}",
    }


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
    
    This function estimates the probability distribution of results
    using Monte Carlo simulations.
    
    Args:
        config_name: Name of the configuration to use.
            Must be one of the predefined configurations.
        input_value: Primary input parameter (0.0 to 10.0).
            {Description of what this parameter represents}
        num_simulations: Number of Monte Carlo iterations (default: 1000).
            More simulations provide more stable uncertainty estimates.
        use_package: Whether to use {PACKAGE_NAME} if available (default: True).
    
    Returns:
        Dict containing:
            - config: Configuration name
            - mean_result: Expected value
            - std_result: Standard deviation
            - ci_lower, ci_upper: 95% confidence interval bounds
            - using_package: Whether external package was used
    
    Example:
        >>> result = calculate_{domain}_result("Config 1", 2.5, 1000)
        >>> print(f"Expected result: {result['mean_result']:.2f}")
    
    Raises:
        ValueError: If config_name is not in CONFIGURATIONS.
    """
    if config_name not in CONFIGURATIONS:
        available = ", ".join(CONFIGURATIONS.keys())
        raise ValueError(f"Unknown config: {config_name}. Available: {available}")
    
    config = CONFIGURATIONS[config_name]
    
    # Calculate results
    if PACKAGE_AVAILABLE and use_package:
        results = _calculate_with_package(config, input_value, num_simulations)
        using_package = True
    else:
        results = _calculate_simplified(config, input_value, num_simulations)
        using_package = False
    
    return {
        "config": config_name,
        "input_value": input_value,
        "mean_result": float(np.mean(results)),
        "std_result": float(np.std(results)),
        "ci_lower": float(np.percentile(results, 2.5)),
        "ci_upper": float(np.percentile(results, 97.5)),
        "using_package": using_package,
        "num_simulations": num_simulations,
    }


def _calculate_with_package(
    config: Dict[str, Any],
    input_value: float,
    num_simulations: int
) -> np.ndarray:
    """Calculate using the external package."""
    # Use the imported package functions here
    # results = package_function(config, input_value, num_simulations)
    raise NotImplementedError("Implement package-specific calculation")


def _calculate_simplified(
    config: Dict[str, Any],
    input_value: float,
    num_simulations: int
) -> np.ndarray:
    """Simplified calculation when package not available."""
    rng = np.random.default_rng()
    
    # Simple model with uncertainty
    base_result = input_value * config["param_a"]
    uncertainty = 0.1 * base_result  # 10% uncertainty
    
    results = rng.normal(base_result, uncertainty, size=num_simulations)
    return np.maximum(0, results)  # Ensure non-negative


# ============================================================================
# Visualization
# ============================================================================

def create_result_histogram(result: Dict[str, Any]) -> plt.Figure:
    """Create histogram of simulation results."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Generate sample data for visualization
    mean = result["mean_result"]
    std = result["std_result"]
    samples = np.random.normal(mean, std, 1000)
    
    ax.hist(samples, bins=50, color='steelblue', alpha=0.7, edgecolor='white')
    ax.axvline(mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean:.2f}')
    ax.axvline(result["ci_lower"], color='orange', linestyle=':', label=f'95% CI')
    ax.axvline(result["ci_upper"], color='orange', linestyle=':')
    
    ax.set_xlabel("Result Value")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Result Distribution - {result['config']}")
    ax.legend()
    
    plt.tight_layout()
    return fig
```

---

## Template 2: Science Module with External Data

Use this template for packages that require downloading and processing external data files.

```python
"""
{Domain} Analysis Module with External Data

This module provides a TerraAgent wrapper for the {PACKAGE_NAME} framework.
{Brief description}

Installation:
    pip install git+{github_url}

Data Source:
    {data_source_description}
    URL: {data_url}

Reference: 
    {Paper citation}
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import warnings
import json

# ============================================================================
# Package and Data Availability
# ============================================================================

PACKAGE_AVAILABLE = False
DATA_READY = False

try:
    from {package} import {main_function}, {process_function}
    PACKAGE_AVAILABLE = True
except ImportError:
    warnings.warn(
        "{PACKAGE_NAME} package not installed. Install with:\n"
        "  pip install git+{github_url}"
    )

# Data directory for processed files
DATA_DIR = Path(__file__).parent.parent / "data" / "{domain}"

# Global data storage (loaded once)
_PROCESSED_DATA = None
_DATA_CONFIG = None


# ============================================================================
# Data Management
# ============================================================================

def _load_data():
    """Load processed data if available."""
    global _PROCESSED_DATA, _DATA_CONFIG, DATA_READY
    
    if not PACKAGE_AVAILABLE:
        return False
    
    data_file = DATA_DIR / "processed" / "data.pqt"
    config_file = DATA_DIR / "processed" / "config.json"
    
    if data_file.exists() and config_file.exists():
        try:
            _PROCESSED_DATA = pd.read_parquet(data_file)
            with open(config_file, "r") as fp:
                _DATA_CONFIG = json.load(fp)
            DATA_READY = True
            return True
        except Exception as e:
            warnings.warn(f"Failed to load data: {e}")
            return False
    
    return False


def setup_data(raw_data_path: Optional[str] = None) -> bool:
    """
    Setup required data files.
    
    Downloads and processes data for use with the framework.
    This only needs to be run once.
    
    Args:
        raw_data_path: Path to raw data file (optional).
            If not provided, will attempt to download.
    
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
    
    # Download data if not provided
    if raw_data_path is None:
        try:
            import urllib.request
            
            # Primary source: Zenodo
            url = "{data_download_url}"
            raw_data_path = DATA_DIR / "raw_data.csv"
            
            print(f"Downloading data from {url}...")
            urllib.request.urlretrieve(url, raw_data_path)
            print(f"Downloaded to {raw_data_path}")
        except Exception as e:
            print(f"Failed to download data: {e}")
            return False
    
    # Process the data using package function
    try:
        print("Processing data...")
        {process_function}(str(DATA_DIR), str(DATA_DIR))
        DATA_READY = True
        print("Data setup complete!")
        return _load_data()
    except Exception as e:
        print(f"Failed to process data: {e}")
        return False


# Try to load data on module import
_load_data()


# ============================================================================
# Status and Configuration
# ============================================================================

LOCATIONS = {
    "Location A": {"lat": 40.0, "lon": -75.0, "param": 1.0},
    "Location B": {"lat": 35.0, "lon": -80.0, "param": 2.0},
}


def get_locations() -> List[str]:
    """Return list of available locations."""
    return list(LOCATIONS.keys())


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
    
    return {
        "installed": PACKAGE_AVAILABLE,
        "data_ready": DATA_READY,
        "package": "{package_name}",
        "install_command": "pip install git+{github_url}",
        "documentation": "{github_url}",
        "data_dir": str(DATA_DIR),
        "mode": mode,
    }


# ============================================================================
# Main Calculation Function
# ============================================================================

def calculate_{domain}_result(
    location_name: str,
    input_value: float,
    num_simulations: int = 1000,
    option_a: str = "default",
    option_b: int = 1,
    use_package: bool = True
) -> Dict[str, Any]:
    """
    Calculate {domain} result with uncertainty quantification.
    
    Uses Monte Carlo simulation with the {PACKAGE_NAME} framework when available,
    or falls back to simplified estimation.
    
    Args:
        location_name: Name of the location to analyze.
            Must be one of the predefined locations.
        input_value: Primary input parameter (0.0 to 10.0).
        num_simulations: Number of Monte Carlo iterations (default: 1000).
        option_a: Configuration option A.
        option_b: Configuration option B (1 or 2).
        use_package: Whether to use {PACKAGE_NAME} if available (default: True).
    
    Returns:
        Dict containing:
            - location: Location name
            - lat, lon: Coordinates
            - mean_result: Expected value
            - std_result: Standard deviation  
            - ci_lower, ci_upper: 95% confidence interval
            - using_package: Whether external package was used
            - mode: Calculation mode used
    """
    if location_name not in LOCATIONS:
        available = ", ".join(LOCATIONS.keys())
        raise ValueError(f"Unknown location: {location_name}. Available: {available}")
    
    loc_data = LOCATIONS[location_name]
    
    # Select calculation method based on availability
    if PACKAGE_AVAILABLE and DATA_READY and use_package:
        results = _calculate_with_package_direct(
            loc_data, input_value, option_a, option_b, num_simulations
        )
        using_package = True
        mode = "direct"
    elif PACKAGE_AVAILABLE and use_package:
        results = _calculate_with_package_style(
            loc_data, input_value, option_a, option_b, num_simulations
        )
        using_package = True
        mode = "style"
    else:
        results = _calculate_simplified(
            loc_data, input_value, option_a, option_b, num_simulations
        )
        using_package = False
        mode = "simplified"
    
    return {
        "location": location_name,
        "lat": loc_data["lat"],
        "lon": loc_data["lon"],
        "input_value": input_value,
        "option_a": option_a,
        "option_b": option_b,
        "mean_result": float(np.mean(results)),
        "std_result": float(np.std(results)),
        "ci_lower": float(np.percentile(results, 2.5)),
        "ci_upper": float(np.percentile(results, 97.5)),
        "using_package": using_package,
        "mode": mode,
        "num_simulations": num_simulations,
    }


def _calculate_with_package_direct(
    loc_data: Dict, input_value: float, 
    option_a: str, option_b: int, num_simulations: int
) -> np.ndarray:
    """Calculate using package with processed data files."""
    # Use _PROCESSED_DATA and _DATA_CONFIG
    # Call package main_function directly
    raise NotImplementedError()


def _calculate_with_package_style(
    loc_data: Dict, input_value: float,
    option_a: str, option_b: int, num_simulations: int
) -> np.ndarray:
    """Calculate using package methodology but without data files."""
    raise NotImplementedError()


def _calculate_simplified(
    loc_data: Dict, input_value: float,
    option_a: str, option_b: int, num_simulations: int
) -> np.ndarray:
    """Simplified calculation when package not available."""
    rng = np.random.default_rng()
    base = input_value * loc_data["param"]
    return rng.normal(base, base * 0.1, size=num_simulations)


# ============================================================================
# Visualization
# ============================================================================

def create_result_histogram(result: Dict[str, Any]) -> plt.Figure:
    """Create histogram of results with statistics."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Regenerate sample distribution for visualization
    rng = np.random.default_rng(42)
    samples = rng.normal(result["mean_result"], result["std_result"], 1000)
    
    ax.hist(samples, bins=50, color='steelblue', alpha=0.7, edgecolor='white')
    ax.axvline(result["mean_result"], color='red', linestyle='--', 
               linewidth=2, label=f'Mean: {result["mean_result"]:.2f}')
    ax.axvline(result["ci_lower"], color='orange', linestyle=':', 
               linewidth=2, label=f'95% CI')
    ax.axvline(result["ci_upper"], color='orange', linestyle=':', linewidth=2)
    
    ax.set_xlabel("Result Value")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Result Distribution - {result['location']} (Mode: {result['mode']})")
    ax.legend()
    
    # Add text box with statistics
    textstr = f"Mode: {result['mode']}\n"
    textstr += f"Package: {'Yes' if result['using_package'] else 'No'}"
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    return fig
```

---

## Template 3: Streamlit App Generation

Use this template for generating Streamlit interfaces from science modules.

```python
"""
Generated Streamlit App for {Domain} Analysis

Auto-generated by TerraAgent from src/science_{domain}.py
"""

import streamlit as st
import pandas as pd
import sys

# Add source directory to path
sys.path.insert(0, "{repo_path}")

from src.science_{domain} import (
    calculate_{domain}_result,
    get_locations,
    check_installation,
    create_result_histogram
)

# Page configuration
st.set_page_config(
    page_title="{Domain} Analysis",
    page_icon="🔬",
    layout="wide"
)

# Header
st.title("🔬 {Domain} Analysis Tool")
st.markdown("""
This application uses the **{PACKAGE_NAME}** framework for {domain} analysis
with uncertainty quantification.
""")

# Check installation status
status = check_installation()
if not status["installed"]:
    st.warning(f"⚠️ {status['package']} not installed. Using simplified mode.")
    st.code(status["install_command"])
elif not status.get("data_ready", True):
    st.info(f"ℹ️ Data not ready. Using '{status['mode']}' mode.")

# Sidebar inputs
st.sidebar.header("📊 Parameters")

location = st.sidebar.selectbox(
    "Location",
    options=get_locations(),
    help="Select a location for analysis"
)

input_value = st.sidebar.slider(
    "Input Value",
    min_value=0.0,
    max_value=10.0,
    value=2.0,
    step=0.1,
    help="Primary input parameter"
)

num_simulations = st.sidebar.slider(
    "Number of Simulations",
    min_value=100,
    max_value=10000,
    value=1000,
    step=100,
    help="More simulations = more stable estimates"
)

# Advanced options
with st.sidebar.expander("Advanced Options"):
    option_a = st.selectbox("Option A", ["default", "alternative"])
    option_b = st.selectbox("Option B", [1, 2])
    use_package = st.checkbox("Use External Package", value=True)

# Run button
if st.sidebar.button("🚀 Run Analysis", type="primary", use_container_width=True):
    with st.spinner("Running analysis..."):
        result = calculate_{domain}_result(
            location_name=location,
            input_value=input_value,
            num_simulations=num_simulations,
            option_a=option_a,
            option_b=option_b,
            use_package=use_package
        )
        st.session_state.result = result

# Display results
if "result" in st.session_state and st.session_state.result:
    result = st.session_state.result
    
    # Map display (if coordinates available)
    if "lat" in result and "lon" in result:
        st.subheader("📍 Location")
        map_df = pd.DataFrame({
            "lat": [result["lat"]],
            "lon": [result["lon"]]
        })
        st.map(map_df, zoom=8)
    
    # Metrics
    st.subheader("📈 Results")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Mean Result", f"{result['mean_result']:.2f}")
    with col2:
        st.metric("Std Deviation", f"{result['std_result']:.2f}")
    with col3:
        st.metric("95% CI", f"{result['ci_lower']:.2f} - {result['ci_upper']:.2f}")
    
    # Status
    st.info(f"Mode: **{result['mode']}** | Package: **{'Yes' if result['using_package'] else 'No'}**")
    
    # Histogram
    st.subheader("📊 Distribution")
    fig = create_result_histogram(result)
    st.pyplot(fig)
    
    # Raw data
    with st.expander("View Raw Data"):
        st.json(result)

else:
    st.info("👆 Configure parameters and click 'Run Analysis' to see results.")
```

---

## Template 4: Rich Streamlit App with Full Visualizations

This is the recommended template for production-quality apps with maps, charts, and sensitivity analysis.

```python
"""
Auto-generated Streamlit UI for {domain} analysis
Generated by TerraAgent IntegratorAgent
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import science module (paste or import)
from src.science_{domain} import (
    calculate_{domain}_result,
    check_installation,
    get_locations,
    LOCATIONS,
    create_histogram
)

# ============================================================================
# Page Configuration
# ============================================================================

st.set_page_config(
    page_title="🌊 {Domain} Analysis Tool",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# Header & Status
# ============================================================================

st.title("🌊 {Domain} Estimator")
st.markdown("""
Estimate {domain} outcomes with uncertainty quantification using Monte Carlo simulation.
""")

# Show package status
status = check_installation()
if status.get('installed') and status.get('data_ready'):
    st.success(f"✅ External package active | Mode: **{status['mode']}**")
elif status.get('installed'):
    st.warning(f"⚠️ Package installed but data not ready | Mode: **{status['mode']}**")
else:
    st.warning(f"⚠️ Using simplified estimation | Install: `{status['install_command']}`")

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

# Location selector
location = st.sidebar.selectbox(
    "📍 Location",
    options=get_locations(),
    help="Select a location for analysis"
)

# Primary parameter with slider
param1 = st.sidebar.slider(
    "🌊 Primary Parameter",
    min_value=0.1,
    max_value=5.0,
    value=1.5,
    step=0.1,
    help="Main input parameter"
)

# Secondary parameters
param2 = st.sidebar.number_input(
    "💰 Secondary Parameter",
    min_value=1000,
    max_value=1000000,
    value=300000,
    step=10000
)

# Category selector with labels
category = st.sidebar.selectbox(
    "🏠 Category",
    options=["A", "B", "C"],
    format_func=lambda x: {"A": "Option A", "B": "Option B", "C": "Option C"}[x],
    help="Select category type"
)

# Simulations slider
num_simulations = st.sidebar.slider(
    "🎲 Monte Carlo Simulations",
    min_value=500,
    max_value=10000,
    value=2000,
    step=500,
    help="More simulations = more accurate uncertainty estimates"
)

st.sidebar.divider()

# Run button
run_clicked = st.sidebar.button(
    "🚀 Run Analysis",
    type="primary",
    use_container_width=True
)

# ============================================================================
# Main Computation
# ============================================================================

if run_clicked:
    with st.spinner("Running Monte Carlo simulation..."):
        try:
            result = calculate_{domain}_result(
                location_name=location,
                param1=param1,
                param2=param2,
                category=category,
                num_simulations=num_simulations
            )
            st.session_state.result = result
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.session_state.result = None

# ============================================================================
# Results Display
# ============================================================================

if st.session_state.result is not None:
    result = st.session_state.result
    
    # Two-column layout
    col1, col2 = st.columns([1, 1])
    
    # ---- Column 1: Metrics & Details ----
    with col1:
        st.subheader("📊 Analysis Results")
        
        # Key metrics row
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Mean Result", f"${result['mean_result']:,.0f}")
        with m2:
            st.metric("Result Ratio", f"{result['ratio']:.1%}")
        with m3:
            st.metric("Std Deviation", f"${result['std_result']:,.0f}")
        
        # Confidence interval
        st.info(f"**95% Confidence Interval:** ${result['ci_lower']:,.0f} — ${result['ci_upper']:,.0f}")
        
        # Details table
        st.markdown("#### 📋 Calculation Details")
        details = pd.DataFrame({
            "Parameter": ["Location", "Input Value", "Category", "Simulations", "Using Package"],
            "Value": [
                result['location'],
                f"{result['param1']:.2f}",
                result['category'],
                result['num_simulations'],
                "✅ Yes" if result['using_package'] else "❌ No"
            ]
        })
        st.dataframe(details, hide_index=True, use_container_width=True)
    
    # ---- Column 2: Map ----
    with col2:
        st.subheader("🗺️ Location Map")
        
        if 'lat' in result and 'lon' in result:
            map_df = pd.DataFrame({
                'lat': [result['lat']],
                'lon': [result['lon']]
            })
            st.map(map_df, zoom=10)
            st.caption(f"📍 {result['location']} ({result['lat']:.4f}, {result['lon']:.4f})")
    
    # ---- Visualizations Section ----
    st.divider()
    st.subheader("📈 Distribution Analysis")
    
    viz_col1, viz_col2 = st.columns([1, 1])
    
    with viz_col1:
        # Histogram
        fig = create_histogram(result)
        st.pyplot(fig)
    
    with viz_col2:
        # Category comparison table
        st.markdown("#### 🏠 Category Comparison")
        comparison_data = []
        for cat in ["A", "B", "C"]:
            r = calculate_{domain}_result(
                location_name=result['location'],
                param1=result['param1'],
                category=cat,
                num_simulations=500
            )
            comparison_data.append({
                'Category': {"A": "Option A", "B": "Option B", "C": "Option C"}[cat],
                'Mean': f"${r['mean_result']:,.0f}",
                'Ratio': f"{r['ratio']:.1%}",
                '95% CI': f"${r['ci_lower']:,.0f}-${r['ci_upper']:,.0f}"
            })
        st.dataframe(pd.DataFrame(comparison_data), hide_index=True, use_container_width=True)
    
    # ---- Sensitivity Analysis ----
    st.divider()
    st.subheader("📉 Parameter Sensitivity")
    
    # Calculate at different parameter values
    param_values = [0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0]
    sensitivity_data = []
    
    with st.spinner("Calculating sensitivity..."):
        for pv in param_values:
            r = calculate_{domain}_result(
                location_name=result['location'],
                param1=pv,
                category=result['category'],
                num_simulations=500
            )
            sensitivity_data.append({
                'Parameter': pv,
                'Mean': r['mean_result'],
                'Ratio': r['ratio'] * 100
            })
    
    sens_df = pd.DataFrame(sensitivity_data)
    
    # Create sensitivity chart
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.fill_between(sens_df['Parameter'], 0, sens_df['Mean'], alpha=0.3, color='steelblue')
    ax2.plot(sens_df['Parameter'], sens_df['Mean'], 'o-', color='steelblue', linewidth=2, markersize=8)
    ax2.axvline(x=result['param1'], color='red', linestyle='--', linewidth=2, 
                label=f"Current: {result['param1']}")
    ax2.set_xlabel('Parameter Value', fontsize=12)
    ax2.set_ylabel('Mean Result', fontsize=12)
    ax2.set_title('Parameter Sensitivity Analysis', fontsize=14)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig2)

else:
    # ---- Initial State: Show instructions ----
    st.info("👈 Configure parameters in the sidebar and click **Run Analysis** to start.")
    
    # Show available locations
    st.subheader("📍 Available Locations")
    
    loc_data = []
    for loc in get_locations():
        info = LOCATIONS[loc]
        loc_data.append({
            'Location': loc,
            'Latitude': info.get('lat', 'N/A'),
            'Longitude': info.get('lon', 'N/A')
        })
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.dataframe(pd.DataFrame(loc_data), hide_index=True, use_container_width=True)
    
    with col2:
        # Show all locations on map
        all_locs = pd.DataFrame([
            {'lat': LOCATIONS[loc].get('lat', 0), 'lon': LOCATIONS[loc].get('lon', 0)}
            for loc in get_locations()
            if 'lat' in LOCATIONS[loc] and 'lon' in LOCATIONS[loc]
        ])
        if not all_locs.empty:
            st.map(all_locs, zoom=3)
```

---

## Checklist for New Modules

### Required Functions
- [ ] `check_installation()` - Returns status dict
- [ ] `get_locations()` or `get_configurations()` - Returns list of options
- [ ] `calculate_{domain}_result()` - Main entry point
- [ ] `create_result_histogram()` - Visualization (optional)

### Required Status Keys
- [ ] `installed` - bool
- [ ] `package` - str
- [ ] `install_command` - str
- [ ] `documentation` - str
- [ ] `mode` - str ("direct", "style", or "simplified")
- [ ] `data_ready` - bool (if applicable)

### Required Result Keys
- [ ] `location` or `config` - str
- [ ] `mean_result` - float
- [ ] `std_result` - float
- [ ] `ci_lower`, `ci_upper` - float
- [ ] `using_package` - bool
- [ ] `lat`, `lon` - float (if location-based)

### Documentation
- [ ] Module docstring with installation instructions
- [ ] Function docstrings with Args, Returns, Example
- [ ] Type hints on all parameters
- [ ] Inline comments for complex logic
