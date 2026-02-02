# TerraAgent v3.0

**Intelligent Earth Science SaaS Platform - Agentic Edition**

Transform scientific Python code into interactive Streamlit web applications with a single click. TerraAgent uses an autonomous agent workflow to analyze, verify, and deploy scientific code as accessible web interfaces.

![TerraAgent Demo](docs/images/terraagent_demo.png)

## 🚀 Quick Start

### Option 1: With Claude Code (Recommended)

Claude Code is an agentic AI coding assistant that can automatically integrate new scientific packages.

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/TerraAgent.git
cd TerraAgent

# 2. Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Open in VS Code with Claude Code extension
code .

# 5. Ask Claude Code to integrate a package:
# "Integrate the UNSAFE package from https://github.com/abpoll/unsafe for flood loss estimation"
```

Claude Code will:
- Read the package documentation
- Create a wrapper module (`src/science_flood.py`)
- Generate a Streamlit app with rich visualizations
- Create launcher scripts for easy deployment

### Option 2: Without Claude Code (Manual)

Use the built-in StreamlitBuilder for rule-based app generation:

```bash
# 1. Clone and setup
git clone https://github.com/yourusername/TerraAgent.git
cd TerraAgent
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # macOS/Linux
pip install -r requirements.txt

# 2. Run the main platform
streamlit run streamlit_app.py

# 3. Use the web interface to:
#    - Enter a GitHub repository URL
#    - Add natural language instructions
#    - Generate an interactive app
```

### Option 3: Double-Click Launcher

**Windows:**
```
Double-click: launch_terraagent.bat
```

**macOS/Linux:**
```bash
chmod +x launch_terraagent.sh
./launch_terraagent.sh
```

---

## 📖 Step-by-Step Usage Guide

### Step 1: Install a Scientific Package

TerraAgent can wrap any scientific Python package. For example, to use the UNSAFE flood framework:

```bash
pip install git+https://github.com/abpoll/unsafe
```

### Step 2: Create a Wrapper Module

Create `src/science_flood.py` following the template pattern:

```python
"""
Flood Loss Uncertainty Assessment Module
Wrapper for the UNSAFE framework.
"""

import numpy as np
from typing import Dict, Any

# Check if package is installed
try:
    from unsafe.ddfs import est_hazus_loss
    UNSAFE_AVAILABLE = True
except ImportError:
    UNSAFE_AVAILABLE = False

def check_unsafe_installation() -> Dict[str, Any]:
    """Check installation status."""
    return {
        "installed": UNSAFE_AVAILABLE,
        "data_ready": True,
        "install_command": "pip install git+https://github.com/abpoll/unsafe"
    }

def calculate_flood_loss(location_name: str, flood_depth: float, ...) -> Dict[str, Any]:
    """Calculate flood loss with uncertainty."""
    # Your implementation here
    return {"mean_loss": 12345, "ci_lower": 5000, "ci_upper": 20000, ...}
```

### Step 3: Generate the Streamlit App

```python
from src.agent_builder import StreamlitBuilder
import inspect
from src import science_flood

builder = StreamlitBuilder()
source_code = inspect.getsource(science_flood)
ui_code = builder.generate_ui_code(source_code, "Create a flood loss estimation app")

# Save the generated app
with open("generated_flood_app.py", "w") as f:
    f.write(ui_code)
```

### Step 4: Create Launcher Scripts

**Windows (launch_flood.bat):**
```batch
@echo off
echo Starting Flood Analysis Tool...
if exist ".venv\Scripts\activate.bat" call .venv\Scripts\activate.bat
streamlit run generated_flood_app.py
pause
```

**macOS/Linux (launch_flood.sh):**
```bash
#!/bin/bash
source .venv/bin/activate
streamlit run generated_flood_app.py
```

### Step 5: Run Your App

```bash
streamlit run generated_flood_app.py
# Or double-click launch_flood.bat (Windows)
```

---

## 🎨 Generated App Features

TerraAgent generates production-quality Streamlit apps with:

| Feature | Description |
|---------|-------------|
| 📍 **Location Map** | Interactive map showing analysis location |
| 📊 **Key Metrics** | Mean, CI, damage ratio with `st.metric()` |
| 📈 **Histograms** | Loss/result distribution visualization |
| 📋 **Details Table** | All parameters in formatted table |
| ✅ **Status Indicator** | Shows package installation status |
| 🎛️ **Smart Widgets** | Auto-generated from function signatures |

### Example Generated UI

```
┌─────────────────────────────────────────────────────────────┐
│ 🌊 Flood Loss Estimator                                     │
│ ✅ UNSAFE framework active | Mode: direct                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  🌊 Flood Risk Analysis              │  🗺️ Location Preview │
│  ───────────────────────             │  ┌─────────────────┐ │
│  Selected Location: Houston, TX      │  │   [Map of TX]   │ │
│  Flood Depth: 1.5 meters             │  │       📍        │ │
│  Property Value: $350,000            │  └─────────────────┘ │
│                                      │  📍 Houston, TX      │
│  Click Run Model to calculate...     │                      │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│ [After clicking Run Model]                                  │
│                                                             │
│  📊 Key Results                      │  🗺️ Location         │
│  ┌────────┬────────┬────────┐        │  ┌─────────────────┐ │
│  │Mean    │Damage  │95% CI  │        │  │   [Map zoom]    │ │
│  │$45,230 │12.9%   │$23K-67K│        │  └─────────────────┘ │
│  └────────┴────────┴────────┘        │                      │
│                                      │                      │
│  📈 Loss Distribution                │                      │
│  ┌─────────────────────────┐         │                      │
│  │    [Histogram chart]    │         │                      │
│  └─────────────────────────┘         │                      │
│                                                             │
│  📋 Calculation Details                                     │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Parameter        │ Value                              │   │
│  │ Location         │ Houston, TX                        │   │
│  │ Flood Depth      │ 1.50 m                             │   │
│  │ Property Value   │ $350,000                           │   │
│  │ Using UNSAFE     │ ✅ Yes                             │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

---

## 🤖 Using Claude Code for Integration

Claude Code can automate the entire integration process. Here are example prompts:

### Integrate a New Package

```
"Integrate the xclim package for climate projections. 
Create a wrapper that calculates temperature anomalies 
with uncertainty quantification."
```

### Generate Rich Visualizations

```
"Update the flood app to show the location map 
before clicking Run Model, making the UI more attractive."
```

### Fix Issues

```
"The app shows 'Using simplified estimation' 
even though UNSAFE is installed. Debug and fix."
```

Claude Code will:
1. Analyze the codebase structure
2. Read CLAUDE.md for integration patterns
3. Create or modify wrapper modules
4. Generate and test Streamlit code
5. Update documentation

---

## 📁 Project Structure

```
TerraAgent/
├── streamlit_app.py          # Main entry point
├── launch_terraagent.bat     # Windows launcher
├── launch_terraagent.sh      # Unix launcher
├── environment.yml           # Conda environment
├── requirements.txt          # Pip dependencies
├── CLAUDE.md                 # Instructions for Claude Code
├── src/
│   ├── main_platform_v3.py   # V3 Agentic platform
│   ├── agent_builder.py      # Core UI generation engine
│   ├── utils.py              # Utility functions
│   ├── agents/
│   │   ├── inspector.py      # Repository analysis
│   │   ├── engineer.py       # Environment setup
│   │   ├── designer.py       # UI generation
│   │   └── integrator.py     # Full workflow orchestration
│   ├── science_climate.py    # Climate analysis module
│   ├── science_fire.py       # Fire weather module
│   └── science_flood.py      # Flood loss module (UNSAFE)
├── data/
│   └── unsafe/               # UNSAFE processed data
├── docs/
│   ├── WORKFLOW.md           # Detailed workflow guide
│   └── TEMPLATES.md          # Template examples
├── notebooks/
│   ├── 01_Verify_Science_Logic.ipynb
│   ├── 02_Test_Agent_Builder.ipynb
│   ├── 03_Test_Flood_Module.ipynb
│   ├── 04_Test_Climate_Module.ipynb
│   └── 05_Test_Fire_Module.ipynb
└── generated_*.py            # Generated Streamlit apps
```

---

## 🔧 Configuration

### Environment Variables

Copy `.env.example` to `.env`:

```bash
OPENAI_API_KEY=sk-...           # For LLM-enhanced generation
ANTHROPIC_API_KEY=sk-ant-...    # For Claude models
GOOGLE_API_KEY=AIza...          # For Gemini models
```

### Supported LLM Providers

| Provider | Models | Notes |
|----------|--------|-------|
| **OpenAI** | gpt-4o, gpt-4o-mini | Best for code generation |
| **Anthropic** | claude-3-5-sonnet | Excellent context understanding |
| **Google** | gemini-1.5-pro | Good for long contexts |
| **Ollama** | llama3.1, codellama | Free, local execution |
| **None** | Rule-based | No API key needed |

Without an API key, TerraAgent uses intelligent rule-based generation that maps Python types to Streamlit widgets automatically.

---

## 🧪 Built-in Demos

Test TerraAgent with these pre-built science modules:

| Demo | Module | External Package |
|------|--------|------------------|
| 🌡️ Climate | `science_climate.py` | xclim |
| 🔥 Fire | `science_fire.py` | xclim |
| 🌊 Flood | `science_flood.py` | UNSAFE |

```python
# Test flood module
from src.science_flood import calculate_flood_loss, check_unsafe_installation

print(check_unsafe_installation())
result = calculate_flood_loss("Houston, TX", 1.5, 1000)
print(f"Mean Loss: ${result['mean_loss']:,.0f}")
```

---

## 📚 Reference

### Architecture

TerraAgent v3.0 uses four collaborative agents:

1. **Inspector Agent** - Clones repos, extracts function signatures via AST
2. **Engineer Agent** - Creates venvs, installs dependencies, runs tests
3. **Designer Agent** - Generates Streamlit UI code with visualizations
4. **Integrator Agent** - Orchestrates full workflow for new packages

### External Packages Supported

- [UNSAFE Framework](https://github.com/abpoll/unsafe) - Flood uncertainty quantification
- [xclim](https://xclim.readthedocs.io/) - Climate and fire indices
- [ClimateBench](https://github.com/duncanwp/ClimateBench) - Climate emulation

---

## 📄 License

See [LICENSE](LICENSE) file.

## 🤝 Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

**Made with ❤️ for Earth Scientists**
