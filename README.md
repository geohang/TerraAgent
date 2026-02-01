# TerraAgent v3.0

**Intelligent Earth Science SaaS Platform - Agentic Edition**

Transform scientific Python code into interactive Streamlit web applications with a single click. TerraAgent uses an autonomous agent workflow to analyze, verify, and deploy scientific code as accessible web interfaces.

## Overview

TerraAgent bridges the gap between scientific code and public accessibility. Scientists provide a GitHub repository URL and natural language instructions, and TerraAgent automatically:

1. **Clones and analyzes** the repository structure
2. **Sets up environments** and verifies code execution
3. **Generates** an interactive Streamlit web application

No UI code required from the scientist.

## Features

- **One-Click Deployment**: Transform any scientific Python script into a web app
- **Agentic Workflow**: Three collaborative AI agents handle the entire pipeline
- **Real-time Terminal**: VS Code-like terminal shows agent operations in real-time
- **LLM Integration**: Supports OpenAI GPT-4o and Anthropic Claude 3.5 Sonnet
- **Rule-based Fallback**: Works without API keys using intelligent type-to-widget mapping
- **Built-in Demos**: Climate, Fire Risk, and Flood Loss science modules included

## Architecture

TerraAgent v3.0 uses three collaborative sub-agents:

### Inspector Agent
- Clones GitHub repositories using GitPython
- Traverses file trees to identify Python files
- Extracts function signatures, docstrings, and return types using AST
- Generates `repo_summary.json` with all code metadata

### Engineer Agent
- Creates isolated virtual environments
- Parses and installs dependencies from `requirements.txt`/`pyproject.toml`
- Runs "smoke tests" to verify code execution
- Implements self-correction when errors occur

### Designer Agent
- Generates Streamlit UI from verified function signatures
- Maps Python types to appropriate Streamlit widgets
- Supports both LLM-enhanced and rule-based generation
- Handles output rendering (figures, dataframes, etc.)

## Installation

### Prerequisites

- Python 3.10+
- Conda (Miniconda or Anaconda)

### Quick Start (Windows)

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/TerraAgent.git
   cd TerraAgent
   ```

2. Double-click `launch_terraagent.bat` to:
   - Create the conda environment automatically
   - Install all dependencies
   - Launch the Streamlit application

3. Open http://localhost:8501 in your browser

### Quick Start (Linux/Mac)

```bash
git clone https://github.com/yourusername/TerraAgent.git
cd TerraAgent
chmod +x launch_terraagent.sh
./launch_terraagent.sh
```

### Manual Installation

```bash
# Create conda environment
conda env create -f environment.yml

# Activate environment
conda activate terraagent

# Run the application
streamlit run streamlit_app.py
```

### Using pip

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Usage

### Basic Workflow

1. **Configure API Key** (optional): Enter your OpenAI or Anthropic API key in the sidebar for enhanced generation
2. **Enter GitHub URL**: Paste a GitHub repository URL or use the demo buttons
3. **Add Instructions**: Describe how you want the app to look (e.g., "Create a heatmap with a year slider")
4. **Generate**: Click "Generate App" and watch the agents work in the terminal

### Demo Mode

TerraAgent includes three built-in science modules:

| Demo | Description |
|------|-------------|
| Climate | Global temperature anomaly simulation with emission scenarios |
| Fire Risk | Fire Weather Index calculation with meteorological inputs |
| Flood Loss | Monte Carlo flood loss uncertainty analysis |

Click any demo button in the sidebar to instantly load and generate an app.

### Example Instructions

- "Create a climate simulation with year slider and emission scenario dropdown"
- "Build a fire risk heatmap with temperature and humidity controls"
- "Show flood loss distribution with property value input"

## Project Structure

```
TerraAgent/
├── streamlit_app.py          # Main entry point
├── launch_terraagent.bat     # Windows launcher
├── launch_terraagent.sh      # Unix launcher
├── environment.yml           # Conda environment
├── requirements.txt          # Pip dependencies
├── src/
│   ├── main_platform_v3.py   # V3 Agentic platform
│   ├── agent_builder.py      # Core UI generation engine
│   ├── utils.py              # Utility functions
│   ├── agents/
│   │   ├── inspector.py      # Repository analysis
│   │   ├── engineer.py       # Environment setup
│   │   └── designer.py       # UI generation
│   ├── science_climate.py    # Climate demo module
│   ├── science_fire.py       # Fire risk demo module
│   └── science_flood.py      # Flood loss demo module
└── notebooks/
    ├── 01_Verify_Science_Logic.ipynb
    └── 02_Test_Agent_Builder.ipynb
```

## Configuration

### Environment Variables

Copy `.env.example` to `.env` and configure:

```bash
OPENAI_API_KEY=sk-...      # Optional: OpenAI API key
ANTHROPIC_API_KEY=sk-ant-... # Optional: Anthropic API key
```

### Supported Models

| Provider | Model | Description |
|----------|-------|-------------|
| OpenAI | gpt-4o | Fast, capable general model |
| Anthropic | claude-3.5-sonnet | Excellent code generation |

Without an API key, TerraAgent uses rule-based generation that maps Python types to Streamlit widgets deterministically.

## API Reference

### StreamlitBuilder

The core engine for UI code generation:

```python
from src.agent_builder import StreamlitBuilder

builder = StreamlitBuilder(llm_client=None)  # or pass OpenAI/Anthropic client
analysis = builder.analyze_code(code_string)
generated_code = builder.generate_ui_code(code_string, instruction)
```

### Agents

```python
from src.agents import InspectorAgent, EngineerAgent, DesignerAgent

# Inspector: Analyze repositories
inspector = InspectorAgent(llm_client=None)
summary = inspector.inspect(github_url, session_id)

# Engineer: Verify execution
engineer = EngineerAgent()
result = engineer.verify(repo_path, target_function)

# Designer: Generate UI
designer = DesignerAgent(llm_client=None)
app = designer.design(function_info, repo_path, instruction, working_snippet)
```

## Development

### Running Tests

```bash
# Verify science modules
jupyter notebook notebooks/01_Verify_Science_Logic.ipynb

# Test agent builder
jupyter notebook notebooks/02_Test_Agent_Builder.ipynb
```

### Science Module Structure

Each science module follows a structured format:

```python
def your_function(param1: int, param2: str) -> plt.Figure:
    """
    Description of function.

    Args:
        param1: Description
        param2: Description

    Returns:
        Matplotlib figure with visualization
    """
    # Implementation
    return fig

if __name__ == "__main__":
    fig = your_function(2050, "High")
    fig.savefig("output.png")
```

## Reference Projects

TerraAgent's demo modules are inspired by:

- [ClimateBench](https://github.com/duncanwp/ClimateBench) - Climate model emulation
- [FireWeatherIndex](https://github.com/steidani/FireWeatherIndex) - Fire weather calculations
- [UNSAFE Framework](https://github.com/abpoll/unsafe) - Flood uncertainty quantification

## Tech Stack

- **Framework**: Python 3.10+, Streamlit 1.30+
- **AI Core**: LangChain, OpenAI, Anthropic
- **Scientific**: NumPy, Pandas, Matplotlib, xarray, GeoPandas
- **System Tools**: GitPython, subprocess, venv

## License

See [LICENSE](LICENSE) file.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request
