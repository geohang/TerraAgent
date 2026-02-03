# TerraAgent Skill for Claude Code

**Transform Scientific Python Packages into Interactive Web Applications**

This skill enables Claude Code to automatically convert scientific Python packages into production-ready Streamlit web applications with rich visualizations, maps, and uncertainty quantification.

---

## Installation

### Option 1: Clone the Repository

```bash
git clone https://github.com/geohang/TerraAgent.git
cd TerraAgent
pip install -r requirements.txt
```

### Option 2: Add as Claude Code Skill

Point Claude Code to this repository:

```
Use TerraAgent from https://github.com/geohang/TerraAgent to integrate scientific packages
```

Claude Code will automatically read the CLAUDE.md file and understand the patterns.

---

## What This Skill Does

Given a GitHub repository URL for a scientific Python package, TerraAgent:

1. **Installs** the package via pip
2. **Creates** a wrapper module (`src/science_{domain}.py`) with:
   - Three-tier fallback (direct/style/simplified)
   - Location-based analysis with coordinates
   - Monte Carlo uncertainty quantification
3. **Generates** a Streamlit app with:
   - Interactive maps
   - Key metrics with confidence intervals
   - Distribution histograms
   - Formatted parameter tables
4. **Creates** launcher scripts for easy deployment

---

## Available Commands

### 1. Integrate a New Package

```
Integrate the {package_name} package from {github_url} for {domain} analysis with uncertainty quantification
```

**Example:**
```
Integrate the UNSAFE package from https://github.com/abpoll/unsafe for flood loss estimation with uncertainty quantification
```

### 2. Generate Streamlit App

```
Generate a Streamlit app for src/science_{domain}.py
```

**Example:**
```
Generate a Streamlit app for src/science_flood.py with map visualization
```

### 3. Create Science Module

```
Create a science module for {domain} analysis following TerraAgent patterns
```

**Example:**
```
Create a science module for earthquake analysis using PyGMT
```

---

## Key Patterns

### Science Module Template

Every science module follows this structure:

```python
"""
{Domain} Analysis Module

Wrapper for the {PACKAGE} framework.
Installation: pip install git+{github_url}
"""

import numpy as np
from typing import Dict, Any, List

# Package availability check
PACKAGE_AVAILABLE = False
try:
    from package import main_function
    PACKAGE_AVAILABLE = True
except ImportError:
    pass

# Sample locations with coordinates (for map display)
LOCATIONS = {
    "Location 1": {"lat": 29.76, "lon": -95.37, "param": 1.0},
    "Location 2": {"lat": 25.76, "lon": -80.19, "param": 2.0},
}

def get_locations() -> List[str]:
    return list(LOCATIONS.keys())

def check_installation() -> Dict[str, Any]:
    return {
        "installed": PACKAGE_AVAILABLE,
        "data_ready": True,
        "install_command": "pip install git+{github_url}"
    }

def calculate_result(
    location_name: str,
    param1: float,
    num_simulations: int = 1000
) -> Dict[str, Any]:
    """Main calculation with uncertainty."""
    loc = LOCATIONS[location_name]

    # Monte Carlo simulation
    results = np.random.normal(10000, 2000, num_simulations)

    return {
        "location": location_name,
        "lat": loc["lat"],          # Required for map
        "lon": loc["lon"],          # Required for map
        "mean_result": float(np.mean(results)),
        "ci_lower": float(np.percentile(results, 2.5)),
        "ci_upper": float(np.percentile(results, 97.5)),
    }
```

### Three-Tier Fallback System

1. **Direct**: Package installed + processed data available
2. **Style**: Package installed, uses package methodology
3. **Simplified**: Package not installed, uses statistical approximation

### Required Return Fields

For map visualization, always include:
- `lat`: Latitude coordinate
- `lon`: Longitude coordinate
- `mean_result`: Central estimate
- `ci_lower`: Lower confidence bound (2.5th percentile)
- `ci_upper`: Upper confidence bound (97.5th percentile)

---

## Built-in Examples

| Module | Domain | External Package | Description |
|--------|--------|------------------|-------------|
| `science_flood.py` | Flood | [UNSAFE](https://github.com/abpoll/unsafe) | Property flood loss with uncertainty |
| `science_climate.py` | Climate | [xclim](https://xclim.readthedocs.io/) | Temperature projections |
| `science_fire.py` | Fire | [xclim](https://xclim.readthedocs.io/) | Fire Weather Index |

---

## Streamlit App Features

Generated apps include:

- **Status Indicator**: Shows package availability mode
- **Map Preview**: Displays location BEFORE running model
- **Sidebar Inputs**: All parameters with tooltips
- **Key Metrics**: Mean, std dev, confidence intervals using `st.metric()`
- **Distribution Plot**: Histogram with mean line
- **Details Table**: All calculation parameters
- **JSON Expander**: Full result object

---

## Usage Tips for Claude Code

1. **Read CLAUDE.md first** - Contains complete integration workflow
2. **Follow the template** - Science modules have a specific structure
3. **Include coordinates** - All locations need lat/lon for maps
4. **Test incrementally**:
   ```python
   from src.science_{domain} import check_installation
   print(check_installation())
   ```
5. **Use StreamlitBuilder** - Don't write Streamlit apps manually

---

## File Structure

```
TerraAgent/
├── CLAUDE.md                 # Main instructions for Claude Code
├── skill/
│   ├── SKILL.md             # This file
│   ├── skill.json           # Skill metadata
│   └── examples/            # Example prompts
├── src/
│   ├── agent_builder.py     # StreamlitBuilder class
│   ├── science_flood.py     # Flood analysis (UNSAFE)
│   ├── science_climate.py   # Climate projections (xclim)
│   └── science_fire.py      # Fire weather (xclim)
└── generated_*.py           # Generated Streamlit apps
```

---

## References

- **Main Documentation**: [CLAUDE.md](../CLAUDE.md)
- **UNSAFE Framework**: https://github.com/abpoll/unsafe
- **xclim**: https://xclim.readthedocs.io/
- **Streamlit**: https://docs.streamlit.io/
