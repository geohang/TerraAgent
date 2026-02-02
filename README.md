# TerraAgent v0.1

**Transform Scientific Python Code into Interactive Web Applications**

TerraAgent automatically converts scientific Python packages into production-ready Streamlit web apps with rich visualizations, maps, and uncertainty quantification.

---

## What Does TerraAgent Do?

**Input:** A GitHub repository URL (e.g., `https://github.com/abpoll/unsafe`)

**Output:** A complete Streamlit web application with:
- 📍 Interactive maps
- 📊 Key metrics and confidence intervals  
- 📈 Distribution histograms
- 📋 Formatted parameter tables
- 🚀 One-click launcher scripts

No frontend coding required from scientists.

---

## Quick Start

### 1. Clone and Setup

```bash
git clone https://github.com/yourusername/TerraAgent.git
cd TerraAgent
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux  
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Run TerraAgent

**Option A: Use the Main Platform (Web Interface)**
```bash
streamlit run streamlit_app.py
```
Then open http://localhost:8501 and enter a GitHub URL.

**Option B: Use the Test Notebooks**
```bash
jupyter notebook notebooks/03_Test_Flood_Module.ipynb
```
Run all cells to generate a flood analysis app.

**Option C: Double-Click Launcher (Windows)**
```
Double-click: launch_terraagent.bat
```

---

## Three Ways to Use TerraAgent

### Method 1: With Claude Code (Recommended for New Packages)

Claude Code is an AI coding assistant that can automatically integrate new scientific packages.

**Quick Setup:**
```python
from src.utils import setup_claude_code
setup_claude_code()  # Checks installation and guides you through setup
```

**Or manually install:**
```powershell
# Windows PowerShell
irm https://claude.ai/install.ps1 | iex

# macOS/Linux
curl -fsSL https://claude.ai/install.sh | bash
```

**Usage:**
1. Open the project in VS Code with Claude Code extension
2. Ask Claude Code to integrate a package:

```
"Integrate the UNSAFE package from https://github.com/abpoll/unsafe 
for flood loss estimation with uncertainty quantification"
```

Claude Code will:
- Install the package
- Create a wrapper module (`src/science_flood.py`)
- Generate a Streamlit app with visualizations
- Create launcher scripts

### Method 2: With LLM APIs (Web Interface)

Use the built-in web interface with OpenAI, Anthropic, or other LLM APIs:

```bash
streamlit run streamlit_app.py
```

1. Enter your API key (OpenAI, Anthropic, etc.) in the sidebar
2. Paste a GitHub repository URL
3. Click "Generate App"

The web interface will use your chosen LLM to generate intelligent Streamlit apps.

**Without an API key:** TerraAgent falls back to rule-based generation which still produces functional apps with rich visualizations for supported domains.

### Method 3: Programmatic (Python API)

Use the built-in `StreamlitBuilder` class directly:

```python
from src.agent_builder import StreamlitBuilder
import inspect
from src import science_flood

# Create builder
builder = StreamlitBuilder()

# Get source code
source_code = inspect.getsource(science_flood)

# Generate Streamlit app
ui_code = builder.generate_ui_code(source_code, "Create a flood analysis app")

# Save
with open("generated_flood_app.py", "w") as f:
    f.write(ui_code)
```

Then run:
```bash
streamlit run generated_flood_app.py
```


---

## Built-in Examples

TerraAgent includes three pre-built science modules:

| Module | Domain | External Package | Description |
|--------|--------|------------------|-------------|
| `science_flood.py` | 🌊 Flood | [UNSAFE](https://github.com/abpoll/unsafe) | Property flood loss with uncertainty |
| `science_climate.py` | 🌡️ Climate | [xclim](https://xclim.readthedocs.io/) | Temperature projections |
| `science_fire.py` | 🔥 Fire | [xclim](https://xclim.readthedocs.io/) | Fire Weather Index |

### Test the Flood Module

```python
from src.science_flood import calculate_flood_loss, check_unsafe_installation

# Check if UNSAFE is installed
print(check_unsafe_installation())

# Calculate flood loss
result = calculate_flood_loss("Houston, TX", flood_depth=1.5, num_simulations=1000)
print(f"Mean Loss: ${result['mean_loss']:,.0f}")
print(f"95% CI: ${result['ci_lower']:,.0f} - ${result['ci_upper']:,.0f}")
```

---

## Generated App Features

TerraAgent generates apps with rich visualizations:

```
┌──────────────────────────────────────────────────────────────┐
│ 🌊 Flood Loss Estimator                                      │
│ ✅ UNSAFE framework active | Mode: direct                    │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  🌊 Flood Risk Analysis          │  🗺️ Location Preview     │
│  ─────────────────────           │  ┌───────────────────┐   │
│  Location: Houston, TX           │  │                   │   │
│  Flood Depth: 1.5 m              │  │   [Interactive    │   │
│  Property Value: $350,000        │  │      Map]         │   │
│                                  │  │       📍          │   │
│  Click Run Model to calculate    │  └───────────────────┘   │
│                                                              │
├──────────────────────────────────────────────────────────────┤
│  [After clicking Run Model]                                  │
│                                                              │
│  📊 Key Results                                              │
│  ┌──────────┬──────────┬──────────┐                          │
│  │ Mean     │ Damage   │ 95% CI   │                          │
│  │ $45,230  │ 12.9%    │ $23K-67K │                          │
│  └──────────┴──────────┴──────────┘                          │
│                                                              │
│  📈 Loss Distribution                                        │
│  [Histogram with mean line]                                  │
│                                                              │
│  📋 Calculation Details                                      │
│  [Formatted table with all parameters]                       │
└──────────────────────────────────────────────────────────────┘
```

**Key UI Features:**
- Map preview shown **before** running (not just after)
- Status indicator shows package availability
- Metrics displayed with `st.metric()` 
- Confidence intervals highlighted
- Expandable JSON for full results

---

## Project Structure

```
TerraAgent/
├── streamlit_app.py           # Main web interface
├── launch_terraagent.bat      # Windows launcher
├── launch_terraagent.sh       # Unix launcher
├── requirements.txt           # Dependencies
├── CLAUDE.md                  # Instructions for Claude Code
│
├── src/
│   ├── agent_builder.py       # StreamlitBuilder class
│   ├── science_flood.py       # Flood analysis (UNSAFE)
│   ├── science_climate.py     # Climate projections (xclim)
│   ├── science_fire.py        # Fire weather (xclim)
│   └── agents/
│       ├── inspector.py       # Repository analysis
│       ├── engineer.py        # Environment setup
│       ├── designer.py        # UI generation
│       └── integrator.py      # Workflow orchestration
│
├── notebooks/
│   ├── 03_Test_Flood_Module.ipynb
│   ├── 04_Test_Climate_Module.ipynb
│   └── 05_Test_Fire_Module.ipynb
│
├── data/
│   └── unsafe/                # Processed HAZUS data
│
└── generated_*.py             # Generated Streamlit apps
```

---

## Creating Your Own Science Module

Follow this pattern to wrap any scientific package:

```python
# src/science_yourpackage.py

"""Wrapper for YourPackage"""

from typing import Dict, Any
import numpy as np

# Check if package is installed
try:
    from yourpackage import main_function
    PACKAGE_AVAILABLE = True
except ImportError:
    PACKAGE_AVAILABLE = False

# Sample locations with coordinates (for map display)
LOCATIONS = {
    "Location A": {"lat": 29.76, "lon": -95.37, "param": 1.0},
    "Location B": {"lat": 25.76, "lon": -80.19, "param": 2.0},
}

def get_locations():
    return list(LOCATIONS.keys())

def check_installation() -> Dict[str, Any]:
    return {
        "installed": PACKAGE_AVAILABLE,
        "data_ready": True,
        "install_command": "pip install yourpackage"
    }

def calculate_result(
    location_name: str,
    param1: float,
    num_simulations: int = 1000
) -> Dict[str, Any]:
    """Main calculation function."""
    loc = LOCATIONS[location_name]
    
    # Your calculation here...
    results = np.random.normal(10000, 2000, num_simulations)
    
    return {
        "location": location_name,
        "lat": loc["lat"],           # Required for map
        "lon": loc["lon"],           # Required for map
        "mean_result": float(np.mean(results)),
        "ci_lower": float(np.percentile(results, 2.5)),
        "ci_upper": float(np.percentile(results, 97.5)),
    }
```

Then generate the app:
```python
from src.agent_builder import StreamlitBuilder
builder = StreamlitBuilder()
code = builder.generate_ui_code(source_code, "Create analysis app")
```

---

## Configuration (Optional)

For LLM-enhanced code generation, set environment variables:

```bash
export OPENAI_API_KEY=sk-...        # OpenAI
export ANTHROPIC_API_KEY=sk-ant-... # Anthropic Claude
```

Without API keys, TerraAgent uses intelligent rule-based generation that works well for most cases.

---

## References

- **UNSAFE Framework**: https://github.com/abpoll/unsafe
- **xclim**: https://xclim.readthedocs.io/
- **Streamlit**: https://docs.streamlit.io/

---

## License

See [LICENSE](LICENSE) file.

---

**Made with ❤️ for Earth Scientists**
