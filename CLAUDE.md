# CLAUDE.md - TerraAgent Instructions for Claude Code

## Overview

TerraAgent is an agentic platform that transforms scientific Python packages into interactive Streamlit web applications. This document provides instructions for Claude Code agents to understand, iterate, and generate new scientific tool integrations.

## Quick Reference

```
GitHub Link → pip install → Create Wrapper Module → Generate Streamlit App → Create Launchers
```

**Key Files:**
- `src/science_*.py` - Wrapper modules for scientific packages
- `src/agent_builder.py` - StreamlitBuilder for UI generation  
- `src/agents/integrator.py` - Full workflow orchestration
- `generated_*_app.py` - Generated Streamlit applications

---

## Complete Integration Workflow

### Step 1: Install the External Package

```bash
pip install git+https://github.com/username/package
```

### Step 2: Create Wrapper Module

Create `src/science_{domain}.py`:

```python
"""
{Domain} Analysis Module

Wrapper for the {PACKAGE} framework.
Installation: pip install git+{github_url}
Reference: {citation}
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from pathlib import Path
import warnings

# ============================================================================
# Package Availability Check
# ============================================================================

PACKAGE_AVAILABLE = False
DATA_READY = False

try:
    from package import main_function
    PACKAGE_AVAILABLE = True
except ImportError:
    warnings.warn("Package not installed. pip install git+{github_url}")


def _find_data_dir():
    """Find data directory - works from both src/ and project root."""
    candidates = [
        Path(__file__).parent.parent / "data" / "{domain}",  # From src/
        Path(__file__).parent / "data" / "{domain}",         # From root
        Path.cwd() / "data" / "{domain}",                    # Current dir
    ]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]

DATA_DIR = _find_data_dir()


# ============================================================================
# Status Functions
# ============================================================================

def check_{domain}_installation() -> Dict[str, Any]:
    """Check installation status."""
    return {
        "installed": PACKAGE_AVAILABLE,
        "data_ready": DATA_READY,
        "package": "{package_name}",
        "install_command": "pip install git+{github_url}",
        "documentation": "{docs_url}",
        "data_dir": str(DATA_DIR),
        "mode": "direct" if DATA_READY else ("style" if PACKAGE_AVAILABLE else "simplified")
    }


# ============================================================================
# Sample Locations/Configurations
# ============================================================================

LOCATIONS = {
    "Location 1": {"lat": 29.76, "lon": -95.37, "param": 1.0},
    "Location 2": {"lat": 25.76, "lon": -80.19, "param": 2.0},
}

def get_locations() -> List[str]:
    """Return available locations."""
    return list(LOCATIONS.keys())


# ============================================================================
# Main Calculation Function  
# ============================================================================

def calculate_{domain}_result(
    location_name: str,
    param1: float,
    num_simulations: int = 1000,
    use_package: bool = True
) -> Dict[str, Any]:
    """
    Main calculation with uncertainty quantification.
    
    Args:
        location_name: Configuration name from LOCATIONS
        param1: Primary parameter (e.g., flood depth, temperature)
        num_simulations: Monte Carlo iterations
        use_package: Use external package if available
    
    Returns:
        Dict with: location, lat, lon, mean_result, ci_lower, ci_upper, 
                   std_result, using_package
    """
    if location_name not in LOCATIONS:
        raise ValueError(f"Unknown: {location_name}. Available: {list(LOCATIONS.keys())}")
    
    loc = LOCATIONS[location_name]
    
    # Three-tier calculation
    if PACKAGE_AVAILABLE and DATA_READY and use_package:
        result = _calculate_direct(loc, param1, num_simulations)
        using_package = True
    elif PACKAGE_AVAILABLE and use_package:
        result = _calculate_style(loc, param1, num_simulations)
        using_package = True
    else:
        result = _calculate_simplified(loc, param1, num_simulations)
        using_package = False
    
    return {
        "location": location_name,
        "lat": loc["lat"],
        "lon": loc["lon"],
        "mean_result": float(np.mean(result)),
        "std_result": float(np.std(result)),
        "ci_lower": float(np.percentile(result, 2.5)),
        "ci_upper": float(np.percentile(result, 97.5)),
        "num_simulations": num_simulations,
        "using_package": using_package,
    }
```

### Step 3: Generate Streamlit App

```python
from src.agent_builder import StreamlitBuilder
import inspect
from src import science_{domain}

builder = StreamlitBuilder()
source = inspect.getsource(science_{domain})
ui_code = builder.generate_ui_code(source, "Create {domain} analysis app")

with open(f"generated_{domain}_app.py", "w") as f:
    f.write(ui_code)
```

### Step 4: Create Launcher Scripts

```python
domain = "{domain}"

# Windows
with open(f"launch_{domain}.bat", 'w') as f:
    f.write(f'''@echo off
echo Starting {domain.title()} Analysis Tool...
if exist ".venv\\Scripts\\activate.bat" call .venv\\Scripts\\activate.bat
streamlit run generated_{domain}_app.py
pause
''')

# Unix
with open(f"launch_{domain}.sh", 'w') as f:
    f.write(f'''#!/bin/bash
source .venv/bin/activate 2>/dev/null || source venv/bin/activate
streamlit run generated_{domain}_app.py
''')
```

---

## Rich Streamlit App Template

Generated apps should include rich visualizations. Here's the complete pattern:

```python
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Include the science module code here
{science_module_code}

# ============================================================================
# Page Configuration
# ============================================================================

st.set_page_config(
    page_title="🌊 {Domain} Analysis",
    page_icon="🌊",
    layout="wide"
)

# ============================================================================
# Header & Status
# ============================================================================

st.title("🌊 {Domain} Estimator")

status = check_{domain}_installation()
if status['installed'] and status['data_ready']:
    st.success(f"✅ Package active | Mode: **{status['mode']}**")
else:
    st.warning(f"⚠️ Simplified mode | Install: `{status['install_command']}`")

# ============================================================================
# Session State
# ============================================================================

if 'result' not in st.session_state:
    st.session_state.result = None

# ============================================================================
# Sidebar Inputs
# ============================================================================

st.sidebar.header("📊 Input Parameters")

location_name = st.sidebar.selectbox(
    "📍 Location", 
    options=get_locations(),
    help="Select analysis location"
)

param1 = st.sidebar.slider(
    "🌊 Parameter 1", 
    min_value=0.1, max_value=5.0, value=1.5, step=0.1
)

num_simulations = st.sidebar.slider(
    "🎲 Simulations", 
    min_value=500, max_value=10000, value=2000, step=500
)

st.sidebar.divider()

# ============================================================================
# IMPORTANT: Show Preview BEFORE Run Button
# This makes the UI more attractive when first loaded
# ============================================================================

st.divider()

preview_col1, preview_col2 = st.columns([2, 1])

with preview_col1:
    st.subheader("🌊 {Domain} Analysis")
    
    st.markdown(f"""
    **Selected Location:** {location_name}  
    **Parameter:** {param1:.1f}  
    **Simulations:** {num_simulations:,}
    
    Click **Run Model** to calculate with uncertainty quantification.
    """)

with preview_col2:
    st.subheader("🗺️ Location Preview")
    # Show selected location on map BEFORE running
    if location_name in LOCATIONS:
        loc_info = LOCATIONS[location_name]
        preview_map = pd.DataFrame({
            'lat': [loc_info['lat']],
            'lon': [loc_info['lon']],
        })
        st.map(preview_map, zoom=8)
        st.caption(f"📍 {location_name}")

st.divider()

# ============================================================================
# Run Button
# ============================================================================

if st.sidebar.button("🚀 Run Model", type="primary", use_container_width=True):
    with st.spinner("Running Monte Carlo simulation..."):
        try:
            result = calculate_{domain}_result(
                location_name=location_name,
                param1=param1,
                num_simulations=num_simulations
            )
            st.session_state.result = result
            st.success("✅ Model completed!")
        except Exception as e:
            st.error(f"Error: {str(e)}")

# ============================================================================
# Results Display (only after running)
# ============================================================================

if st.session_state.result is not None:
    result = st.session_state.result
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📊 Key Results")
        
        # Metrics row
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Mean", f"${result['mean_result']:,.0f}")
        with m2:
            st.metric("Std Dev", f"${result['std_result']:,.0f}")
        with m3:
            mode = "✅ Full" if result['using_package'] else "📊 Simplified"
            st.metric("Mode", mode)
        
        # Confidence interval
        st.info(f"**95% CI:** ${result['ci_lower']:,.0f} — ${result['ci_upper']:,.0f}")
        
        # Histogram
        st.markdown("#### 📈 Distribution")
        fig, ax = plt.subplots(figsize=(8, 4))
        mean = result['mean_result']
        std = result['std_result']
        samples = np.random.normal(mean, std, 1000)
        samples = np.maximum(samples, 0)
        ax.hist(samples, bins=30, edgecolor='white', alpha=0.7, color='steelblue')
        ax.axvline(mean, color='red', linestyle='--', linewidth=2, label='Mean')
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    
    with col2:
        st.subheader("🗺️ Location")
        map_df = pd.DataFrame({
            'lat': [result['lat']],
            'lon': [result['lon']],
        })
        st.map(map_df, zoom=10)
        st.caption(f"📍 {result['location']}")
    
    # Details table
    st.divider()
    st.subheader("📋 Calculation Details")
    
    details = pd.DataFrame([
        {"Parameter": "Location", "Value": result['location']},
        {"Parameter": "Simulations", "Value": f"{result['num_simulations']:,}"},
        {"Parameter": "Using Package", "Value": "✅ Yes" if result['using_package'] else "❌ No"},
    ])
    st.dataframe(details, hide_index=True, width="stretch")
    
    # Full JSON
    with st.expander("🔍 View Full JSON"):
        st.json(result)
```

---

## Key Patterns

### 1. Three-Tier Fallback

Always implement:
- **Direct**: Full package with processed data
- **Style**: Package installed, methodology similar
- **Simplified**: Fallback when package unavailable

### 2. Flexible Data Directory

Use `_find_data_dir()` to locate data from both `src/` and project root:

```python
def _find_data_dir():
    candidates = [
        Path(__file__).parent.parent / "data" / "domain",
        Path(__file__).parent / "data" / "domain",
        Path.cwd() / "data" / "domain",
    ]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]
```

### 3. Show Preview Before Run

**Critical for UX:** Display the map and parameters summary BEFORE the user clicks Run:

```python
# Preview section BEFORE run button
preview_col1, preview_col2 = st.columns([2, 1])

with preview_col1:
    st.subheader("🌊 Analysis")
    st.markdown(f"**Location:** {location_name}")

with preview_col2:
    st.subheader("🗺️ Preview")
    st.map(preview_df, zoom=8)

# THEN the run button and results
if st.sidebar.button("🚀 Run"):
    ...
```

### 4. Status Check Function

Every module needs `check_{domain}_installation()`:

```python
def check_{domain}_installation() -> Dict[str, Any]:
    return {
        "installed": PACKAGE_AVAILABLE,
        "data_ready": DATA_READY,
        "package": "package_name",
        "install_command": "pip install git+...",
        "mode": "direct" | "style" | "simplified"
    }
```

### 5. Return Dict with Coordinates

For map visualization, always include `lat` and `lon`:

```python
return {
    "location": location_name,
    "lat": 29.76,
    "lon": -95.37,
    "mean_result": float(np.mean(result)),
    ...
}
```

---

## Streamlit Widget Mapping

| Python Type | Streamlit Widget |
|-------------|-----------------|
| `str` (location) | `st.selectbox` with `get_locations()` |
| `str` (free text) | `st.text_input` |
| `float` | `st.slider` or `st.number_input` |
| `int` (small) | `st.slider` |
| `int` (large) | `st.number_input` |
| `bool` | `st.checkbox` |
| `Literal[...]` | `st.selectbox` with options |

---

## Visualization Components

| Component | Widget | When to Use |
|-----------|--------|-------------|
| Key metrics | `st.metric()` | Mean, CI, ratios |
| Confidence interval | `st.info()` | Range display |
| Map | `st.map()` | When lat/lon available |
| Histogram | `st.pyplot(fig)` | Distribution |
| Table | `st.dataframe()` | Parameters, comparisons |
| JSON | `st.json()` in expander | Full result details |

---

## Claude Code Agent Rules

### When Integrating a New Package:

1. **READ** the package README and examples
2. **CREATE** `src/science_{domain}.py` using the template
3. **TEST** with `check_installation()` and main function
4. **GENERATE** Streamlit app with `StreamlitBuilder`
5. **CREATE** launcher scripts (.bat and .sh)
6. **UPDATE** HISTORY.md with integration details

### When Fixing Issues:

1. **CHECK** `check_installation()` output
2. **VERIFY** `_find_data_dir()` returns correct path
3. **TEST** main function in terminal
4. **REGENERATE** app if needed

### When Improving UI:

1. **ADD** preview section before Run button
2. **INCLUDE** status indicator at top
3. **USE** `st.metric()` for key results
4. **ADD** `st.map()` when coordinates available
5. **CREATE** histogram for distributions

---

## Testing Checklist

```python
# 1. Check installation
from src.science_{domain} import check_{domain}_installation
print(check_{domain}_installation())

# 2. Test main function
from src.science_{domain} import calculate_{domain}_result
result = calculate_{domain}_result("Location 1", 1.5, 1000)
print(result)

# 3. Verify Streamlit code
from src.agent_builder import StreamlitBuilder
import inspect
from src import science_{domain}
builder = StreamlitBuilder()
code = builder.generate_ui_code(inspect.getsource(science_{domain}), "Test")
compile(code, '<test>', 'exec')  # Syntax check
print(f"Generated {len(code)} characters")

# 4. Run app
# streamlit run generated_{domain}_app.py
```

---

## References

- **UNSAFE Framework**: https://github.com/abpoll/unsafe
- **xclim**: https://xclim.readthedocs.io/
- **Streamlit Docs**: https://docs.streamlit.io/
- **TerraAgent Templates**: docs/TEMPLATES.md
