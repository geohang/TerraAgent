🌍 Project TerraAgent: Earth Science Code-to-App Agent

1. Project Vision

Goal: Build a "Text-to-App Platform for Earth Science."
Core Value: Bridge the gap between scientific code and public accessibility. Scientists simply upload a raw Python script (Climate, Fire, Flood), and TerraAgent automatically understands the logic and generates an interactive Streamlit Web Interface, without the scientist writing a single line of UI code.

2. Reference Architectures & Data Sources

We do not clone these projects directly. Instead, we reference their I/O logic and scientific methodology as the core material for our Demos.

🌤️ Climate Change

Reference Project: ClimateBench https://github.com/duncanwp/ClimateBench

Core Logic: Machine learning simulation for climate models. Input CO2 concentration/year, output global temperature anomaly heatmap.

Keywords: xarray, NetCDF, Temperature Anomaly, Heatmap.

🔥 Wildfire Risk 

Reference Project: Fire Weather Index (FWI) https://www.google.com/search?q=https://github.com/steidani/FireWeatherIndex


Core Logic: Calculate FWI based on meteorological data (temperature, humidity, wind speed, rain) and generate a risk distribution map.

Keywords: numpy, Meteorology, Risk Map, Spatial Grid.

🌊 Flood Loss Assessment

Reference Project: UNSAFE Framework https://github.com/abpoll/unsafe

Core Logic: Flood risk assessment based on probability theory. Uses Monte Carlo Simulation to calculate uncertainty, rather than a single deterministic prediction.

Keywords: Uncertainty Quantification, Probabilistic Modeling, Damage Curves, Histogram.

3. Tech Stack

Core Language: Python 3.10+

UI Engine: Streamlit (v1.30+)

AI Orchestration: LangChain + OpenAI (GPT-4o) OR Anthropic (Claude 3.5 Sonnet)

Scientific Libraries:

xarray, numpy, pandas, scipy (Data Processing)

matplotlib, seaborn (Static Plotting)

geopandas (Geospatial Data)

Dev Tools: Jupyter Notebooks (for testing and verification)

4. Phase 1: Mocking the Scientific Core

Goal: Create three independent Python files that mimic the logic of the Reference Projects above. These will serve as "test cases" for the Agent.

Instructions for AI Assistant: Generate the following three files. They must strictly follow the Structured Code Standard below to ensure the Agent can parse them correctly.

Structured Code Standard:

Imports: All imports must be at the top.

Type Hints: Every function argument and return value must have Python type hints.

Docstrings: Use Google-style docstrings describing args and returns.

Main Guard: Include an if __name__ == "__main__": block that runs a simple test of the function and saves the plot to a file (e.g., output.png).

A. science_climate.py (Mocking ClimateBench)

Function Signature: simulate_warming(year: int, emission_scenario: str) -> plt.Figure

Logic:

Grid Generation: Use numpy to generate a global lat/lon grid (180x360).

Physics Simulation: Calculate warming magnitude based on input year (delta from present) and emission_scenario (modifiers: "Low"=0.5x, "High"=1.5x).

Polar Amplification: Multiply latitude factor to make poles warm 2x faster than equator.

Output: A Matplotlib Figure of the Global Temperature Anomaly Heatmap with a proper colorbar.

B. science_fire.py (Mocking FWI System)

Function Signature: calculate_fwi_risk(temperature: float, humidity: int, wind_speed: float, rain_mm: float) -> plt.Figure

Logic:

Spatial Grid: Create a 50x50 simulation region grid representing a forest.

Index Calculation: Apply a simplified FWI formula: $Risk = (Temp \times 0.4) - (Humidity \times 0.2) + (Wind \times 0.5) - (Rain \times 2)$.

Terrain Noise: Add Perlin-like noise or random variation to simulate elevation effects.

Output: A Risk Heatmap using a 'YlOrRd' (Yellow-Orange-Red) colormap.

C. science_flood.py (Mocking UNSAFE Framework)

Function Signature: calculate_loss_uncertainty(flood_depth: float, property_value: int, num_simulations: int = 5000) -> plt.Figure

Logic:

Monte Carlo Setup: Initialize arrays for 5000 iterations.

Damage Function: Define a sigmoid-like depth-damage curve where damage % increases with depth.

Uncertainty Injection: Add Gaussian noise to the damage curve for each iteration (simulating structural differences).

Financial Calc: $Loss = PropertyValue \times (DamageRatio + Noise)$.

Output: A Probability Distribution Histogram showing the Mean loss and 95% confidence interval vertical lines.

5. Phase 2: The Builder Agent

Goal: Write the core engine that reads the Python code above and automatically writes the Streamlit App.

File Name: agent_builder.py

Class Structure:

class StreamlitBuilder:

def __init__(self, llm_client): Initialize LLM connection.

def analyze_code(self, code_str: str) -> dict: Use inspect or Regex to extract function signature, docstring, and return type.

def generate_ui_code(self, code_str: str) -> str: The main orchestration method.

System Prompt Strategy:

"You are a senior Streamlit Developer. I will give you a Python function signature. Please write a complete Streamlit application for it.

Code Structure Rules:

Imports: Import streamlit as st, matplotlib.pyplot as plt, and the necessary scientific libraries.

State Management: Use st.session_state to prevent re-running the model on every interaction.

Layout: Use st.sidebar for all input parameters (int -> sliders, str -> selectbox). Use the main area for the Output.

Action: Crucial! Include a st.button('Run Model'). Only call the science function when clicked.

Output Handling: Detect the return type. If plt.Figure, use st.pyplot(). If pd.DataFrame, use st.dataframe().

6. Phase 3: TerraAgent Unified Platform

Goal: A Universal "Code-to-App" Loader that works for ANY script, including our demos.

File Name: main_platform.py

App Layout Structure:

Header: Title "TerraAgent Platform" and description.

Sidebar - Quick Start:

A section "Load Demo".

Buttons: [Load Climate], [Load Fire], [Load Flood].

Action: Clicking a button loads the content of science_*.py into the Main Input Area.

Main Content - Split View:

Column 1 (Input): st.text_area ("Python Code Input") where users can paste code or see the loaded demo code.

Action Button: "Build & Visualize App".

Column 2 (Preview): The "App Sandbox".

When "Build" is clicked, pass code to agent_builder.

Receive generated Streamlit code string.

Advanced: Use exec() to execute the generated Streamlit code context inside the current app (using a container) or simply display the generated code for now if exec is too complex for Phase 1.

7. Phase 4: Notebooks for Testing & Demos

Goal: Provide interactive notebooks to verify the scientific logic and the agent's generation capabilities before deploying the app.

Notebook A: notebooks/01_Verify_Science_Logic.ipynb

Purpose: Ensure the mock science files produce valid plots without errors.

Structure:

Import science_climate, science_fire, science_flood.

Cell 1: Run simulate_warming(2050, "High") and display the plot inline.

Cell 2: Run calculate_fwi_risk(...) with various inputs to check sensitivity.

Cell 3: Run the Monte Carlo simulation and verify the histogram distribution looks Gaussian/Log-normal.

Notebook B: notebooks/02_Test_Agent_Builder.ipynb

Purpose: Test the LLM prompt and code generation without the Web UI.

Structure:

Initialize agent_builder.

Load the text of science_climate.py.

Run builder.generate_ui_code(code_text).

Output: Print the generated Python code string.

Manual Verification: Check if the generated code includes st.sidebar, st.button, and correct st.pyplot calls.

8. Execution Plan (Prompts for Claude Code / Cursor)

Please execute the following Prompts in order using your AI coding assistant (Claude Code CLI or Cursor):

Step 1: Environment Setup

"Initialize the project structure for 'TerraAgent'. Create a virtual environment and a requirements.txt containing: streamlit, openai, langchain, xarray, pandas, numpy, matplotlib, seaborn, scipy. Install them."

Step 2: Generate Scientific Cores

"Generate the three science core files (science_climate.py, science_fire.py, science_flood.py) based on the 'Phase 1' specs in the Blueprint. Make sure to include the if __name__ == '__main__': blocks for local testing."

Step 3: Generate Notebooks

"Create the folder notebooks/. Generate 01_Verify_Science_Logic.ipynb to import and run the three science functions. Generate 02_Test_Agent_Builder.ipynb to mock the agent generation process."

Step 4: Develop Agent Engine

"Develop agent_builder.py following the Class Structure in Phase 2. Ensure it properly mocks the LLM response if no API key is present, so we can test the flow."

Step 5: Platform Integration

"Build main_platform.py with the 2-column layout defined in Phase 3. Connect the 'Quick Load' buttons to file readers that populate the text area."




