# TerraAgent Workflow Guide

This document describes the complete workflow for integrating external scientific Python packages into TerraAgent.

## Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         TerraAgent Agentic Workflow                          │
└─────────────────────────────────────────────────────────────────────────────┘

    ┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
    │   GitHub     │     │   Install    │     │   Generate   │     │   Create     │
    │   Link       │ ──► │   Package    │ ──► │   Wrapper    │ ──► │   Interface  │
    └──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
           │                    │                    │                    │
           ▼                    ▼                    ▼                    ▼
      Inspector            Engineer             Integrator            Designer
        Agent               Agent                 Agent                Agent
```

## Phase 1: Repository Analysis (Inspector Agent)

### Input
- GitHub repository URL (e.g., `https://github.com/abpoll/unsafe`)

### Process
1. Clone the repository (shallow clone for efficiency)
2. Traverse the file tree to find Python files
3. Parse AST to extract function signatures
4. Identify main entry points (functions with `calculate_`, `run_`, `process_` prefixes)
5. Extract docstrings and type hints
6. Identify data requirements from README and examples

### Output
```python
{
    "repo_url": "https://github.com/abpoll/unsafe",
    "package_name": "unsafe",
    "main_functions": [
        {
            "name": "est_hazus_loss",
            "file": "src/unsafe/ddfs.py",
            "signature": "(ddf_types, depths, ffes, ddfs, MAX_DICT, base_adj)",
            "docstring": "...",
            "return_type": "pd.Series"
        }
    ],
    "data_requirements": [
        {
            "file": "haz_fl_dept.csv",
            "source": "zenodo",
            "url": "https://zenodo.org/records/10027236"
        }
    ],
    "dependencies": ["numpy", "pandas", "geopandas"]
}
```

## Phase 2: Package Installation (Engineer Agent)

### Input
- Repository analysis from Phase 1

### Process
1. Install the package from GitHub:
   ```bash
   pip install git+https://github.com/{owner}/{repo}
   ```

2. Install additional dependencies if needed:
   ```bash
   pip install geopandas rasterio pyyaml
   ```

3. Verify installation by importing:
   ```python
   from unsafe.ddfs import est_hazus_loss, process_hazus
   ```

4. Run smoke tests on main functions

### Output
```python
{
    "success": True,
    "package": "unsafe",
    "version": "0.2.1",
    "installed_modules": ["unsafe.ddfs", "unsafe.ensemble", "unsafe.const"],
    "verified_functions": ["est_hazus_loss", "process_hazus"],
    "errors": []
}
```

## Phase 3: Wrapper Module Generation (Integrator Agent)

### Input
- Verified functions from Phase 2
- Data requirements from Phase 1

### Process
1. Create data directory structure:
   ```
   data/{domain}/
   └── physical/
       ├── hazus_ddfs.pqt
       └── hazus.json
   ```

2. Implement data download function:
   ```python
   def setup_data():
       """Download from Zenodo/GitHub and process."""
       import urllib.request
       url = "https://zenodo.org/records/{id}/files/{file}?download=1"
       urllib.request.urlretrieve(url, local_path)
       process_hazus(raw_dir, output_dir)
   ```

3. Create wrapper module with three-tier fallback:
   ```python
   if PACKAGE_AVAILABLE and DATA_READY:
       return _calculate_direct(...)  # Use package directly
   elif PACKAGE_AVAILABLE:
       return _calculate_style(...)   # Similar methodology
   else:
       return _calculate_simplified(...)  # Basic fallback
   ```

4. Add standard interfaces:
   - `check_installation()` - Returns status dict
   - `setup_data()` - Downloads and processes data
   - `get_locations()` - Returns demo configurations
   - `calculate_{domain}_result()` - Main entry point

### Output
- Complete `src/science_{domain}.py` module
- Populated `data/{domain}/` directory

## Phase 4: Interface Generation (Designer Agent)

### Input
- Wrapper module from Phase 3
- User requirements (e.g., "Create flood loss calculator with map")

### Process
1. Analyze function signature:
   ```python
   func_info = builder.analyze_code(source_code)
   ```

2. Generate Streamlit widgets for each parameter:
   - `str` with options → `st.selectbox`
   - `float` → `st.slider` or `st.number_input`
   - `int` → `st.slider`
   - `bool` → `st.checkbox`

3. Generate output handling:
   - Dict with lat/lon → `st.map()`
   - Dict with metrics → `st.metric()`
   - plt.Figure → `st.pyplot()`
   - pd.DataFrame → `st.dataframe()`

4. Add state management and action button

### Output
- Complete Streamlit application code
- Saved to `generated_{domain}_app.py`

## Complete Example: UNSAFE Integration

### Step 1: User Input
```
GitHub URL: https://github.com/abpoll/unsafe
Requirements: "Create flood loss calculator with uncertainty visualization"
```

### Step 2: Inspector Analysis
```python
repo_summary = inspector.analyze_repository("https://github.com/abpoll/unsafe")
# Found: est_hazus_loss, process_hazus, est_naccs_loss
# Data: haz_fl_dept.csv from Zenodo
```

### Step 3: Engineer Installation
```python
result = engineer.install_github_package("https://github.com/abpoll/unsafe")
# Installed: unsafe 0.2.1
# Verified: from unsafe.ddfs import est_hazus_loss ✓
```

### Step 4: Integrator Wrapper
```python
module = integrator.create_wrapper_module(
    package="unsafe",
    domain="flood",
    main_function="est_hazus_loss",
    data_source="zenodo:10027236"
)
# Created: src/science_flood.py
# Downloaded: data/unsafe/haz_fl_dept.csv
# Processed: data/unsafe/physical/hazus_ddfs.pqt
```

### Step 5: Designer Interface
```python
app_code = designer.generate_streamlit_app(
    module_path="src/science_flood.py",
    user_instruction="Create flood loss calculator with uncertainty"
)
# Generated: Streamlit app with sidebar inputs, map, histogram
```

### Step 6: Result
User gets a working Streamlit app that:
- Uses UNSAFE's `est_hazus_loss()` directly
- Shows flood loss with uncertainty distribution
- Displays location on map
- Provides 95% confidence intervals

## Automation Commands

### Full Pipeline
```python
from src.agents import IntegratorAgent

integrator = IntegratorAgent()
result = integrator.integrate_package(
    github_url="https://github.com/abpoll/unsafe",
    domain="flood",
    user_instruction="Create flood loss calculator"
)
# Returns path to generated Streamlit app
```

### Step-by-Step
```python
from src.agents import InspectorAgent, EngineerAgent, DesignerAgent

# 1. Analyze
inspector = InspectorAgent()
analysis = inspector.analyze_repository(github_url)

# 2. Install
engineer = EngineerAgent()
install_result = engineer.install_github_package(github_url)

# 3. Generate wrapper (manual for complex packages)
# ... create science_{domain}.py following templates

# 4. Create interface
designer = DesignerAgent()
app_code = designer.generate_app(source_code, instruction)
```

## Data Sources

### Zenodo Downloads
```python
import urllib.request
url = f"https://zenodo.org/records/{record_id}/files/{filename}?download=1"
urllib.request.urlretrieve(url, local_path)
```

### GitHub Raw Files
```python
url = f"https://raw.githubusercontent.com/{owner}/{repo}/main/{path}"
urllib.request.urlretrieve(url, local_path)
```

### Processing Functions
Most scientific packages provide processing functions:
```python
# UNSAFE example
from unsafe.ddfs import process_hazus
process_hazus(raw_data_dir, output_dir)
```

## Error Handling

### Installation Failures
```python
try:
    pip install git+{url}
except:
    # Try without optional dependencies
    pip install git+{url} --no-deps
    pip install required_deps_only
```

### Data Download Failures
```python
urls = [
    "https://zenodo.org/records/...",  # Primary
    "https://raw.githubusercontent.com/...",  # Fallback
]
for url in urls:
    try:
        download(url)
        break
    except:
        continue
```

### Import Failures
```python
try:
    from package import function
    AVAILABLE = True
except ImportError:
    AVAILABLE = False
    # Use fallback implementation
```

## Testing Checklist

- [ ] Package installs successfully
- [ ] Main functions are importable
- [ ] Data downloads without errors
- [ ] Data processing completes
- [ ] `check_installation()` returns correct status
- [ ] Main function returns expected structure
- [ ] Streamlit code compiles
- [ ] Generated app runs without errors
