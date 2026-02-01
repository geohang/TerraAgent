roject TerraAgent: Earth Science Code-to-App Platform (v2.0)

1. Project Vision

Goal: Build a "SaaS-like Platform" for Earth Science Visualization.
Core Workflow:

Source: Scientist provides a GitHub URL (Python script).

Intent: Scientist provides a Natural Language Instruction (e.g., "Make the map red and add a slider for years").

Result: TerraAgent generates and renders an interactive Streamlit App instantly.

2. Tech Stack

Core Language: Python 3.10+

UI Engine: Streamlit (v1.30+)

AI Orchestration: LangChain + OpenAI (GPT-4o) / Claude 3.5 Sonnet

Utilities: requests (for fetching raw GitHub code)

Scientific Libraries: xarray, numpy, pandas, scipy, matplotlib, seaborn, geopandas

3. Architecture Overview

3.1 The Input Layer (The Source)

Instead of local files, we now support Remote GitHub Files.

Raw Code Fetcher: A utility that converts github.com/blob/... links to raw.githubusercontent.com/... and downloads the Python logic.

3.2 The Intelligence Layer (The Agent)

Context: The Agent reads the entire scientific script to understand functions, classes, and data structures.

Instruction Injection: The Agent accepts a user_instruction string to customize the UI (e.g., styling, layout preferences).

Code Generation: Outputs a standalone app.py script.

3.3 The Presentation Layer (The UI)

Split View: Left Panel (Config/Code) vs. Right Panel (Live Preview).

Dynamic Runtime: Uses Python's exec() to render the generated code safely within the user's session.

4. Phase 1: The Demo Ecosystem (Hosted Demos)

To verify the platform, we use three "Golden Standard" scripts. In a real scenario, these would be hosted on a public GitHub repo, but for development, we mock them locally or upload them to a Gist.

ClimateBench: simulate_warming(year, scenario) -> Heatmap.

FWI (Fire): calculate_fwi_risk(temp, humidity, wind) -> Risk Map.

UNSAFE (Flood): calculate_loss_uncertainty(depth, value) -> Histogram.

5. Phase 2: Agent Engine Logic (agent_builder.py)

The Builder is the brain. It must be updated to handle the new input flow.

Function Signature:

def generate_ui_code(self, code_str: str, user_instruction: str) -> str:


System Prompt Strategy:

"You are an expert Streamlit Developer.
Inputs:

Scientific Code: {code_str}

User Instruction: "{user_instruction}"

Task:
Write a streamlit app that imports/defines the necessary logic from the scientific code and builds a UI.

Rules:

Instruction Adherence: If the user asks for 'Red colors', 'Sidebar layout', or 'Specific titles', YOU MUST COMPLY.

Self-Contained: Do not import the scientific code as a module. COPY the necessary functions into your generated code so it runs standalone.

Execution: Always include a 'Run' button to trigger heavy calculations.

Robustness: Use try-except blocks for data loading."

6. Phase 3: Platform UI Specification (main_platform.py)

This describes the exact layout implemented in the code.

Sidebar (Configuration):

API Key Input (Password type).

Model Selector (GPT-4o / Claude).

"Quick Start" Buttons (Load Demo URLs).

Main Layout (Split Columns):

Left Column (Input):

GitHub File URL (Text Input).

Natural Language Instruction (Text Area).

Generate App (Primary Button).

Right Column (Preview):

Tab 1 (App): Live rendering of the generated tool.

Tab 2 (Code): Source code display with a "Download .py" button.

7. Execution Plan (Prompts for Cursor/Claude)

Use these prompts to build or verify the project components.

Step 1: Utilities

"Create utils.py. Implement fetch_github_code(url) to handle GitHub raw URL conversion and fetching. Add validate_api_key(key)."

Step 2: Agent Logic

"Update agent_builder.py. Ensure generate_ui_code takes a user_instruction argument. Modify the LangChain prompt to explicitly prioritize the user's natural language requests for UI customization."

Step 3: Platform UI

"Update main_platform.py. Implement the Split-View layout.

Left: URL input + Chat box.

Right: st.tabs(['Preview', 'Code']).

Logic: When 'Generate' is clicked, fetch code -> call agent -> exec() the result."

Step 4: Integration Test

"Run the app. Paste a raw link to a simple Python script (e.g., a sine wave generator). Enter instruction: 'Use a slider for frequency and make the line green'. Verify the result."