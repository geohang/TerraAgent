# TerraAgent Development History

## Project Final Target

**TerraAgent v3.0: Intelligent Earth Science SaaS Platform (Agentic Edition)**

### Core Vision
"One-Click Science to SaaS" - Instantly transform scientific code into SaaS services.

### Target Workflow
1. **Input**: Scientist provides GitHub Repo Link + Natural Language Requirements + LLM API Key
2. **Process**: TerraAgent's autonomous agents handle environment configuration, dependency installation, code comprehension, and interface testing
3. **Output**: Automatically deploys and renders an interactive Streamlit Web App

### Three Collaborative Agents
- **Inspector Agent**: Clones repos, traverses file trees, identifies entry points, extracts function signatures
- **Engineer Agent**: Creates virtual environments, installs dependencies, runs smoke tests, self-corrects errors
- **Designer Agent**: Generates Streamlit UI from verified code using LLM or rule-based generation

---

## Development History

### Session: 2026-02-01

#### Phase 1: Initial Cleanup & Restructuring

**Files Deleted:**
- `run.py` - Old launcher script (replaced by `launch_terraagent.bat`)
- `Target.md` - v2 specification document (no longer needed)
- `Target2.md` - v3 specification document (no longer needed)
- `notebooks/target2.agent.md` - Leftover spec document
- `src/main_platform.py` - Legacy v2 platform (keeping only v3)
- All `__pycache__/` directories - Python bytecode cache

**Files Created:**
- `README.md` - Comprehensive project documentation
- `.gitignore` - Git ignore rules for Python, venv, IDE files, etc.

**Files Updated:**
- `environment.yml` - Added missing `anthropic` and `gitpython` packages
- `requirements.txt` - Added missing dependencies
- `streamlit_app.py` - Updated to use v3 platform

**Commit:** `026f2d7` - "Refactor to v3.0 agentic architecture with cleanup"

---

#### Phase 2: Demo System Overhaul

**Problem Identified:**
The demo system used local mock functions instead of demonstrating the real agentic workflow (cloning from GitHub).

**Solution Implemented:**

1. **Added GitHub Examples** - Real repository demos:
   - ClimateBench: `https://github.com/duncanwp/ClimateBench`
   - FireWeatherIndex: `https://github.com/steidani/FireWeatherIndex`
   - UNSAFE Flood: `https://github.com/abpoll/unsafe`

2. **Kept Local Demos** - Built-in science modules for offline testing:
   - `src/science_climate.py`
   - `src/science_fire.py`
   - `src/science_flood.py`

---

#### Phase 3: Multi-Provider LLM Support

**Problem Identified:**
Only OpenAI and Anthropic were supported. Users wanted more LLM options.

**Solution Implemented:**

Added support for 6 LLM providers:

| Provider | Description |
|----------|-------------|
| **OpenAI** | GPT-4o, o1, and other OpenAI models |
| **Anthropic** | Claude 4, Claude 3.5, and other Anthropic models |
| **Google** | Gemini 2.0, Gemini 1.5, and other Google AI models |
| **Mistral** | Mistral Large, Codestral, and other Mistral models |
| **OpenRouter** | Aggregator for 100+ models from various providers |
| **Ollama (Local)** | Local LLMs via Ollama (Llama, CodeLlama, etc.) |

**Files Modified:**
- `src/main_platform_v3.py` - Added `LLM_PROVIDERS` config and updated `get_llm_client()`
- `src/agent_builder.py` - Added `_call_llm()` method to handle all providers

---

#### Phase 4: Flexible Model Version Input

**Problem Identified:**
Hardcoded model dropdown lists became outdated quickly as LLM providers release new models frequently.

**Solution Implemented:**

Changed from dropdown selection to text input for model names:
- Users can now type **any model version** they want to use
- Default values provided as sensible starting points
- Placeholder text shows example model names
- Help links to each provider's model documentation

| Provider | Default Model | Example Alternatives |
|----------|---------------|---------------------|
| **OpenAI** | gpt-4o | o1-preview, gpt-4-turbo |
| **Anthropic** | claude-sonnet-4-20250514 | claude-opus-4-20250514 |
| **Google** | gemini-2.0-flash | gemini-1.5-pro, gemini-2.0-pro |
| **Mistral** | mistral-large-latest | codestral-latest |
| **OpenRouter** | openai/gpt-4o | anthropic/claude-3.5-sonnet |
| **Ollama** | llama3.2 | codellama, deepseek-coder |

**Files Modified:**
- `src/main_platform_v3.py` - Replaced model dropdown with text input, added docs links
- `HISTORY.md` - Expanded future work roadmap

---

#### Phase 5: Bug Fixes & Map Visualization

**Problem Identified:**
1. Unicode escape error when executing generated code on Windows (backslashes in paths)
2. Flood demo lacked geographic context - users wanted map visualization

**Solution Implemented:**

1. **Fixed Unicode Error** in `agent_builder.py`:
   - Removed problematic double `json.dumps()` calls
   - Properly escape user instruction strings for embedding in generated code
   - CSS block handling moved to conditional markdown output

2. **Enhanced Flood Module** with map support:
   - Added predefined flood-prone US cities with coordinates
   - New `calculate_flood_loss()` function returns Dict with lat/lon
   - Location-adjusted risk factors affect damage calculations
   - Results displayed with `st.map()` showing selected location

3. **Smart Widget Generation**:
   - Auto-detect "location" parameters and generate selectbox
   - Map output detection for Dict returns with coordinates
   - Display metrics (Mean Loss, 95% CI) alongside map

| Location | Risk Factor | Avg Property |
|----------|-------------|--------------|
| Houston, TX | 75% | $350,000 |
| Miami, FL | 80% | $450,000 |
| New Orleans, LA | 85% | $280,000 |
| Cedar Rapids, IA | 65% | $220,000 |
| Sacramento, CA | 55% | $520,000 |

**Files Modified:**
- `src/agent_builder.py` - Fixed string escaping, added map output handling
- `src/science_flood.py` - Rewrote with location data and map coordinates
- `src/main_platform_v3.py` - Updated flood demo instruction for map visualization

---

#### Phase 6: Enhanced LLM Code Generation

**Problem Identified:**
When users provide an LLM API key, the system should use the AI to generate better, more customized Streamlit apps.

**Solution Implemented:**

1. **Improved System Prompt** for LLM generation:
   - Detailed instructions for modern Streamlit features
   - Specific guidance for `st.map()`, `st.metric()`, `st.columns()`
   - Parameter type to widget mapping rules
   - Output type detection (Figure, DataFrame, Dict with lat/lon)

2. **Enhanced User Prompt**:
   - Includes parameter details with descriptions and defaults
   - Clear return type specification
   - Explicit instructions for map visualization when needed
   - Requirement to include full source code in generated app

3. **Better Logging**:
   - Shows which LLM provider/model is being used
   - Displays "🤖 Using LLM" vs "📋 Using rule-based generation"
   - Logs function analysis details (return type, parameters)

4. **Updated Launcher Script**:
   - Checks multiple conda environment locations
   - Works with existing `terraagent` environment
   - Better path detection for Anaconda/Miniconda

**Files Modified:**
- `src/agent_builder.py` - Enhanced SYSTEM_PROMPT and user prompt
- `src/main_platform_v3.py` - Added LLM usage logging
- `launch_terraagent.bat` - Fixed environment detection

---

#### Phase 7: Claude Code CLI Integration

**Problem Identified:**
Users wanted access to Claude's advanced agentic coding capabilities directly in TerraAgent without needing separate API keys.

**Solution Implemented:**

1. **Added Claude Code as LLM Provider**:
   - Claude Code CLI is an agentic coding tool from Anthropic
   - Supports `sonnet`, `opus`, and `haiku` model variants
   - Uses `claude -p "prompt"` for programmatic code generation
   - No API key required (uses authenticated CLI session)

2. **New Provider Option**:
   - Added "Claude Code (CLI)" as first option in provider dropdown
   - Automatically checks if CLI is installed via `claude --version`
   - Falls back gracefully if CLI not available

3. **Implementation Details**:
   - `check_claude_code_installed()` - Verifies CLI availability
   - `call_claude_code(prompt, model, working_dir)` - Subprocess wrapper
   - `_call_claude_code()` in StreamlitBuilder - Code generation method
   - 180-second timeout for complex generation tasks

4. **Installation Instructions**:
   - Windows: `irm https://claude.ai/install.ps1 | iex`
   - macOS/Linux: `curl -fsSL https://claude.ai/install.sh | bash`
   - Documentation: https://docs.anthropic.com/en/docs/claude-code

| Model | Description | Best For |
|-------|-------------|----------|
| sonnet | Fast, balanced | Default choice |
| opus | Most capable | Complex tasks |
| haiku | Fastest | Simple generation |

**Files Modified:**
- `src/main_platform_v3.py` - Added Claude Code provider and utility functions
- `src/agent_builder.py` - Added `_call_claude_code()` method

---

#### Phase 8: Widget Generation Bug Fixes

**Problem Identified:**
Generated Streamlit apps had syntax errors due to:
1. Multi-line widget code with incorrect indentation
2. Integer sliders with fixed max_value=200 that was too small for large defaults like `num_simulations=5000`

**Solution Implemented:**

1. **Fixed Widget Indentation**:
   - Changed location selectbox from multi-line to single-line format
   - Removed extra indentation from sidebar code assembly
   - Widgets now generate at top-level without nested indentation

2. **Smart Integer Widget Selection**:
   - Large integers (>200) now use `number_input` instead of `slider`
   - Slider bounds dynamically calculated based on default value
   - Float sliders also now scale based on default value

3. **Testing Notebook Added**:
   - `notebooks/03_Test_Flood_Demo.ipynb` for debugging
   - Tests all three science modules (Flood, Climate, Fire)
   - Validates generated code syntax before deployment
   - Saves generated app to file for manual testing

**Files Modified:**
- `src/agent_builder.py` - Fixed `_generate_widget()` and sidebar code assembly
- `notebooks/03_Test_Flood_Demo.ipynb` - New testing notebook

---

#### Phase 9: UNSAFE Framework Integration (Complete)

**Problem Identified:**
The flood loss module used custom simplified damage functions. Users wanted integration with the real UNSAFE (UNcertain Structure And Fragility Ensemble) framework from GitHub for more accurate, research-grade flood risk estimation.

**User Request:** "I want you to use https://github.com/abpoll/unsafe directly, follow how they ask for install and example"

**Solution Implemented:**

1. **UNSAFE Package Installation**:
   ```bash
   pip install git+https://github.com/abpoll/unsafe
   ```

2. **HAZUS DDF Data Downloaded from Zenodo**:
   - Source: https://zenodo.org/records/10027236
   - File: `haz_fl_dept.csv` (HAZUS flood depth-damage functions)
   - Citation: Pollack, A. (2023). Flood Depth-Damage Functions from HAZUS

3. **Direct UNSAFE Function Integration**:
   - Uses UNSAFE's `process_hazus()` to generate processed DDF parquet files
   - Calls UNSAFE's `est_hazus_loss()` directly for damage estimation
   - Processed data stored in `data/unsafe/physical/` directory

4. **Data Files Generated by UNSAFE**:
   | File | Description |
   |------|-------------|
   | `hazus_ddfs.pqt` | Depth-damage functions with uncertainty params |
   | `hazus.json` | Maximum depth dictionary for DDF lookup |
   | `hazus_ddfs_nounc.pqt` | DDFs without uncertainty (benchmark) |
   | `hazus_nounc.json` | Max depth dict without uncertainty |

5. **DDF Types Available (HAZUS RES1)**:
   | Code | Description |
   |------|-------------|
   | 1SNB_RES1 | 1-Story, Slab, No Basement |
   | 1SWB_A_RES1 | 1-Story, With Basement, A-Zone |
   | 1SWB_V_RES1 | 1-Story, With Basement, V-Zone |
   | 1SPL_RES1 | 1-Story, Pile Foundation |
   | 2SNB_RES1 | 2-Story, Slab, No Basement |
   | 2SWB_A_RES1 | 2-Story, With Basement, A-Zone |
   | 2SWB_V_RES1 | 2-Story, With Basement, V-Zone |
   | 2SPL_RES1 | 2-Story, Pile Foundation |

6. **Three-Tier Estimation System**:
   - **Direct Mode**: When UNSAFE installed + data ready → calls `est_hazus_loss()` directly
   - **Style Mode**: When UNSAFE installed but data not ready → HAZUS-style curves
   - **Simplified Mode**: When UNSAFE not installed → basic sigmoidal damage curve

7. **Test Results (Direct UNSAFE)**:
   ```
   Mode: direct
   Data Ready: True
   DDF Type: 1SNB_RES1 (1-story, slab, no basement)
   Depth: 4.92 ft (1.5m)
   FFE: 0.75 ft
   Relative Damage: 36.04%
   ```

8. **Helper Functions Added**:
   - `setup_unsafe_data()` - Downloads DDFs from Zenodo and processes with UNSAFE
   - `check_unsafe_installation()` - Returns status dict with mode, data_ready, etc.
   - `_estimate_loss_with_unsafe_direct()` - Wrapper for `est_hazus_loss()`

**Reference:**
- UNSAFE Framework: https://github.com/abpoll/unsafe
- HAZUS DDFs on Zenodo: https://doi.org/10.5281/zenodo.10027236
- Paper: Pollack, A., Doss-Gollin, J., Srikrishnan, V., & Keller, K. (2024)
- DOI: https://doi.org/10.21105/joss.07527

**Files Modified:**
- `src/science_flood.py` - Complete rewrite with direct UNSAFE integration
- `data/unsafe/` - New directory with processed DDF files
- `notebooks/03_Test_Flood_Demo.ipynb` - Added UNSAFE testing cells
- `requirements.txt` - Added note about optional UNSAFE installation

---

## Current Project Structure

```
TerraAgent/
├── streamlit_app.py          # Main entry point (v3)
├── launch_terraagent.bat     # Windows one-click launcher
├── launch_terraagent.sh      # Unix launcher
├── environment.yml           # Conda environment
├── requirements.txt          # Pip dependencies
├── README.md                 # Project documentation
├── HISTORY.md                # This file
├── .gitignore                # Git ignore rules
├── .env.example              # API key template
├── LICENSE
├── src/
│   ├── main_platform_v3.py   # Agentic platform UI
│   ├── agent_builder.py      # Core code generation engine
│   ├── utils.py              # Utility functions
│   ├── __init__.py
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── inspector.py      # Repository analysis agent
│   │   ├── engineer.py       # Environment verification agent
│   │   └── designer.py       # UI generation agent
│   ├── science_climate.py    # Climate simulation demo
│   ├── science_fire.py       # Fire risk assessment demo
│   └── science_flood.py      # Flood loss uncertainty demo
└── notebooks/
    ├── 01_Verify_Science_Logic.ipynb
    └── 02_Test_Agent_Builder.ipynb
```

---

## Key Features Implemented

### 1. Agentic Workflow
- Three-phase pipeline: Code Ingestion → Environment Verification → UI Generation
- Real-time VS Code-like terminal output
- Session management with unique IDs

### 2. Multi-Provider LLM Support
- 7 providers including Claude Code CLI
- Dynamic provider/model selection in sidebar
- Claude Code: Agentic coding without API key
- Automatic fallback to rule-based generation

### 3. Dual Demo System
- **GitHub Examples**: Clone real scientific repositories
- **Local Demos**: Built-in modules for offline testing

### 4. Rule-Based Fallback
- Works without any API key
- Intelligent Python type → Streamlit widget mapping
- Parameter extraction from docstrings

---

## Phase 10: Agentic Workflow Enhancement (Current)

**User Request:** Create a proper agentic workflow:
1. Get GitHub link → 2. Auto install → 3. Generate wrapper module → 4. Create interface

**Solution Implemented:**

### 1. Claude Code Documentation (CLAUDE.md)
Created comprehensive documentation for Claude Code agents to understand and iterate TerraAgent:
- Core workflow explanation
- Module structure guidelines
- Integration patterns (three-tier fallback)
- Testing checklist

### 2. Workflow Guide (docs/WORKFLOW.md)
Detailed step-by-step workflow documentation:
- Phase 1: Repository Analysis (Inspector Agent)
- Phase 2: Package Installation (Engineer Agent)
- Phase 3: Wrapper Module Generation (Integrator Agent)
- Phase 4: Interface Generation (Designer Agent)

### 3. Templates Documentation (docs/TEMPLATES.md)
Comprehensive templates for creating science modules:
- Basic Science Module template
- Science Module with External Data template
- Streamlit App Generation template
- Checklists for required functions and keys

### 4. IntegratorAgent (src/agents/integrator.py)
New orchestrator agent that combines all other agents:
```python
from src.agents import IntegratorAgent

integrator = IntegratorAgent()
result = integrator.integrate_package(
    github_url="https://github.com/owner/package",
    domain="science",
    user_instruction="Create analysis tool"
)
```

Features:
- `integrate_package()` - Full integration pipeline
- `quick_integrate()` - Minimal configuration integration
- `check_integration_status()` - Check existing integrations
- Auto-generates wrapper modules with three-tier fallback
- Handles data file downloads

### 5. EngineerAgent Enhancements
New methods for GitHub package integration:
- `install_github_package()` - Install from git+URL
- `download_data_file()` - Download from Zenodo/GitHub
- `verify_package_import()` - Verify imports after installation

### 6. Updated agents/__init__.py
Export all new classes:
- `IntegratorAgent`, `IntegrationResult`, `PackageAnalysis`
- All data classes now exportable

### 7. Launcher Script Generation
IntegratorAgent now automatically generates launcher scripts after Streamlit app creation:
- **Windows**: `launch_{domain}.bat` - Double-click to start the app
- **Unix**: `launch_{domain}.sh` - Shell script for Linux/Mac
- Auto-detects virtual environments (.venv or venv)
- Shows startup messages and instructions

Example output:
```
🚀 Step 6: Generating launcher script...
  Created Windows launcher: launch_flood.bat
  Created Unix launcher: launch_flood.sh

💡 Double-click the launcher to start the app!
```

### 8. Rich Streamlit App Generation
IntegratorAgent now generates production-quality Streamlit apps with full visualizations:
- **Status Indicators**: Shows package/data availability with colored badges
- **Key Metrics**: Prominent display of mean, std dev, ratios using `st.metric()`
- **Map Visualization**: Interactive map showing selected location
- **Distribution Charts**: Histogram of Monte Carlo simulation results
- **Sensitivity Analysis**: Charts showing parameter sensitivity
- **Comparison Tables**: Side-by-side category/foundation comparisons
- **Details Tables**: Formatted calculation parameters

Updated documentation:
- `CLAUDE.md` - Added Streamlit visualization guidelines and template
- `docs/TEMPLATES.md` - Added Template 4: Rich Streamlit App with Full Visualizations

**Files Created:**
- `CLAUDE.md` - Instructions for Claude Code agents
- `docs/WORKFLOW.md` - Detailed workflow guide
- `docs/TEMPLATES.md` - Module templates
- `src/agents/integrator.py` - New IntegratorAgent

**Files Modified:**
- `src/agents/engineer.py` - Added GitHub install methods
- `src/agents/__init__.py` - Export new classes

---

## Pending / Future Work

### High Priority
- [ ] **Flexible LLM Model Input**: Allow users to type any model version instead of hardcoded dropdown lists
- [ ] **Repository Caching**: Implement caching for cloned repositories to speed up repeated analyses
- [ ] **Private Repository Support**: Add GitHub token authentication for private repos
- [ ] **Error Recovery**: Improve Engineer Agent's self-correction capabilities with retry logic

### Medium Priority
- [ ] **More GitHub Examples**: Add curated scientific repositories from different Earth science domains
- [ ] **Unit Testing**: Comprehensive test suite for all agents (Inspector, Engineer, Designer, Integrator)
- [ ] **Docker Deployment**: Containerized deployment option for production environments
- [ ] **Multi-Output Support**: Support for Plotly, Altair, Bokeh, and other visualization libraries

### Low Priority / Nice-to-Have
- [ ] **Session Persistence**: Save and restore agentic sessions across browser refreshes
- [ ] **Batch Processing**: Process multiple repositories in parallel
- [ ] **Custom Widget Templates**: User-defined Streamlit widget templates for common patterns
- [ ] **Export Options**: Export generated apps as standalone Python packages
- [ ] **Collaboration Features**: Share generated apps with team members
- [ ] **Version Control Integration**: Auto-commit generated code to user's repository

### Research & Exploration
- [ ] **Multi-Agent Orchestration**: Explore LangGraph or CrewAI for more sophisticated agent coordination
- [ ] **Code Understanding**: Integrate tree-sitter for better AST-based code analysis
- [ ] **Fine-tuned Models**: Train specialized models for Earth science code comprehension
- [ ] **Feedback Loop**: Collect user feedback to improve generation quality over time
