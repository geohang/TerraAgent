"""
TerraAgent Builder Engine

This module provides the core engine for automatically generating Streamlit
applications from Python scientific code. It analyzes function signatures,
type hints, and docstrings to create appropriate UI components.

The StreamlitBuilder class can work with or without an LLM backend.
"""

import ast
import json
import re
from typing import Any, Dict, List, Optional, Tuple


class StreamlitBuilder:
    """
    Builder class that reads Python code and generates Streamlit applications.

    This class analyzes Python source code to extract function signatures,
    docstrings, and type hints, then generates corresponding Streamlit UI code.

    Attributes:
        llm_client: Optional LLM client for enhanced code generation.
    """

    # System prompt for LLM-based generation
    SYSTEM_PROMPT = """You are a senior Streamlit Developer specializing in scientific applications. Generate a complete, production-ready Streamlit application.

## Code Structure Rules:

### Imports
- Import streamlit as st, numpy as np, pandas as pd
- Import matplotlib.pyplot as plt if visualization needed
- Include ALL necessary imports at the top

### Page Setup
- Use st.set_page_config(page_title="App Name", layout="wide") at the very beginning
- Add a clear st.title() and st.markdown() description

### State Management  
- Use st.session_state to store results and prevent re-computation
- Initialize with: if 'result' not in st.session_state: st.session_state.result = None

### Input Layout
- Use st.sidebar for ALL input parameters
- int parameters → st.sidebar.slider() with reasonable min/max
- float parameters → st.sidebar.slider() or st.sidebar.number_input()
- str parameters with known options → st.sidebar.selectbox()
- str parameters for location/city → st.sidebar.selectbox() with location list
- bool parameters → st.sidebar.checkbox()

### Action Button
- CRUCIAL: Include st.sidebar.button("Run Model", type="primary", use_container_width=True)
- Only call the science function when the button is clicked
- Use st.spinner() to show loading state

### Output Handling
- plt.Figure → st.pyplot(fig)
- pd.DataFrame → st.dataframe(df)
- Dict with lat/lon → st.map() with pd.DataFrame, plus st.metric() for key values
- Dict without coordinates → st.json(result)
- Numeric results → st.metric() with delta if applicable

### Map Visualization
- If result contains 'lat' and 'lon', display with st.map()
- Create DataFrame: pd.DataFrame({'lat': [result['lat']], 'lon': [result['lon']]})
- Use st.map(df, zoom=10)

### Metrics Display
- Use st.columns() to show multiple metrics side by side
- Format currency: f"${value:,.0f}"
- Format percentages: f"{value:.1%}"

## Output Format
Output ONLY valid Python code. No explanations, no markdown code blocks, no comments outside the code."""

    def __init__(self, llm_client: Optional[Any] = None):
        """
        Initialize the StreamlitBuilder.

        Args:
            llm_client: Optional LLM client (e.g., OpenAI, Anthropic).
                If None, uses rule-based generation.
        """
        self.llm_client = llm_client

    def analyze_code(self, code_str: str) -> Dict[str, Any]:
        """
        Analyze Python code to extract function signature, docstring, and return type.

        Uses AST parsing and regex to extract:
        - Function name
        - Parameters with types and defaults
        - Return type
        - Docstring with parameter descriptions

        Args:
            code_str: Python source code as a string.

        Returns:
            dict: Dictionary containing:
                - 'name': Function name
                - 'docstring': Full docstring text
                - 'parameters': List of parameter dicts
                - 'return_type': Return type annotation
        """
        tree = ast.parse(code_str)

        # Collect all public functions
        public_functions = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and not node.name.startswith('_'):
                public_functions.append(node)
        
        if not public_functions:
            return {}
        
        # Prefer functions with parameters (more likely to be the main entry point)
        # Also prefer functions with "calculate", "run", "process", "analyze" in name
        priority_keywords = ['calculate', 'run', 'process', 'analyze', 'compute', 'simulate']
        
        best_func = public_functions[0]
        best_score = 0
        
        for func in public_functions:
            score = len(func.args.args)  # More params = higher priority
            func_name_lower = func.name.lower()
            for keyword in priority_keywords:
                if keyword in func_name_lower:
                    score += 10  # Boost for priority keywords
                    break
            if score > best_score:
                best_score = score
                best_func = func
        
        return self._extract_function_info(best_func, code_str)

    def _extract_function_info(self, node: ast.FunctionDef, code_str: str) -> Dict[str, Any]:
        """Extract detailed information from a function AST node."""
        docstring = ast.get_docstring(node) or ""

        # Extract parameters
        parameters = []
        defaults_offset = len(node.args.args) - len(node.args.defaults)

        for i, arg in enumerate(node.args.args):
            if arg.arg == 'self':
                continue

            # Get type hint
            type_hint = "Any"
            if arg.annotation:
                type_hint = ast.unparse(arg.annotation)

            # Get default value
            default = None
            default_idx = i - defaults_offset
            if default_idx >= 0 and default_idx < len(node.args.defaults):
                default_node = node.args.defaults[default_idx]
                try:
                    default = ast.literal_eval(ast.unparse(default_node))
                except (ValueError, SyntaxError):
                    default = ast.unparse(default_node)

            # Extract description from docstring
            description = self._get_param_description(arg.arg, docstring)

            # Check for enum options in type hint
            options = None
            if 'Literal' in type_hint:
                options = re.findall(r'"([^"]+)"', type_hint)

            parameters.append({
                'name': arg.arg,
                'type': type_hint,
                'default': default,
                'description': description,
                'options': options
            })

        # Get return type
        return_type = "Any"
        if node.returns:
            return_type = ast.unparse(node.returns)

        return {
            'name': node.name,
            'docstring': docstring,
            'parameters': parameters,
            'return_type': return_type
        }

    def _get_param_description(self, param_name: str, docstring: str) -> str:
        """Extract parameter description from docstring."""
        if not docstring:
            return ""

        pattern = rf'{param_name}:\s*(.+?)(?=\n\s*\w+:|Args:|Returns:|Example:|$)'
        match = re.search(pattern, docstring, re.DOTALL | re.IGNORECASE)

        if match:
            desc = match.group(1).strip()
            # Get first sentence only
            first_line = desc.split('\n')[0].strip()
            return ' '.join(first_line.split())[:150]

        return ""

    def generate_ui_code(self, code_str: str, user_instruction: str = "") -> str:
        """
        Generate Streamlit UI code from Python source code.

        This is the main orchestration method that:
        1. Analyzes the input code
        2. Extracts function information
        3. Generates Streamlit code (via LLM or rules)
        4. Incorporates user natural-language instructions (layout/style hints)

        Args:
            code_str: Python source code as a string.
            user_instruction: Natural-language directions for layout/style.

        Returns:
            str: Generated Streamlit application code.
        """
        # Analyze the code first
        func_info = self.analyze_code(code_str)

        if not func_info:
            raise ValueError("No suitable function found in the provided code")

        # Use LLM if available, otherwise use rule-based generation
        if self.llm_client:
            return self._generate_with_llm(func_info, code_str, user_instruction)
        else:
            return self._generate_rule_based(func_info, code_str, user_instruction)

    def _generate_with_llm(self, func_info: Dict, code_str: str, user_instruction: str) -> str:
        """Generate Streamlit code using LLM (supports multiple providers)."""
        # Build the prompt
        params_str = ', '.join([
            f"{p['name']}: {p['type']}" + (f" = {p['default']}" if p['default'] is not None else "")
            for p in func_info['parameters']
        ])
        signature = f"def {func_info['name']}({params_str}) -> {func_info['return_type']}"
        
        # Build parameter details for better LLM understanding
        param_details = []
        for p in func_info['parameters']:
            detail = f"- {p['name']} ({p['type']})"
            if p.get('description'):
                detail += f": {p['description']}"
            if p.get('default') is not None:
                detail += f" [default: {p['default']}]"
            param_details.append(detail)
        param_details_str = '\n'.join(param_details) if param_details else "No parameters"

        user_prompt = f"""Generate a complete Streamlit app for this scientific function.

## Function Signature
```python
{signature}
```

## Parameters
{param_details_str}

## Return Type
{func_info['return_type']}

## Docstring
{func_info['docstring']}

## User Requirements
{user_instruction}

## Full Source Code (include this in the generated app)
```python
{code_str}
```

## Important Notes
1. Include the FULL source code above in your generated app (don't import it)
2. The generated code must be self-contained and runnable
3. If the return type is Dict with lat/lon, use st.map() for visualization
4. If user mentions "map", add geographic visualization with st.map()
5. Use st.metric() to display key numeric results prominently

Generate the complete Streamlit app code now:"""

        try:
            code = self._call_llm(user_prompt)

            # Extract code from markdown if present
            if "```python" in code:
                code = code.split("```python")[1].split("```")[0]
            elif "```" in code:
                code = code.split("```")[1].split("```")[0]

            return code.strip()

        except Exception as e:
            print(f"LLM generation failed: {e}, falling back to rule-based")
            return self._generate_rule_based(func_info, code_str, user_instruction)

    def _call_llm(self, user_prompt: str) -> str:
        """Call the LLM based on provider type."""
        # Handle new dict-based client format
        if isinstance(self.llm_client, dict):
            client = self.llm_client.get("client")
            model = self.llm_client.get("model")
            provider = self.llm_client.get("provider")

            # Handle Claude Code CLI
            if provider == "claude_code":
                return self._call_claude_code(user_prompt, model)

            if provider == "openai" or provider == "openrouter" or provider == "ollama":
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": self.SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0
                )
                return response.choices[0].message.content

            elif provider == "anthropic":
                response = client.messages.create(
                    model=model,
                    max_tokens=4096,
                    system=self.SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": user_prompt}]
                )
                return response.content[0].text

            elif provider == "google":
                model_instance = client.GenerativeModel(model)
                full_prompt = f"{self.SYSTEM_PROMPT}\n\n{user_prompt}"
                response = model_instance.generate_content(full_prompt)
                return response.text

            elif provider == "mistral":
                response = client.chat.complete(
                    model=model,
                    messages=[
                        {"role": "system", "content": self.SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt}
                    ]
                )
                return response.choices[0].message.content

        # Legacy: Handle direct client objects (backward compatibility)
        elif hasattr(self.llm_client, 'chat'):
            # OpenAI-style API
            response = self.llm_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0
            )
            return response.choices[0].message.content

        elif hasattr(self.llm_client, 'messages'):
            # Anthropic-style API
            response = self.llm_client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=4096,
                system=self.SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}]
            )
            return response.content[0].text

        raise ValueError("Unsupported LLM client type")

    def _call_claude_code(self, user_prompt: str, model: str = "sonnet") -> str:
        """
        Call Claude Code CLI to generate code.
        
        Uses the Claude Code CLI in print mode for non-interactive code generation.
        Claude Code is an agentic coding tool that understands codebases.
        
        Args:
            user_prompt: The full prompt including system instructions
            model: Model alias (sonnet, opus, haiku)
        
        Returns:
            Generated code as string
        """
        import subprocess
        
        # Combine system prompt and user prompt for Claude Code
        full_prompt = f"""{self.SYSTEM_PROMPT}

---

{user_prompt}"""
        
        cmd = [
            "claude",
            "-p",  # Print mode (non-interactive)
            "--model", model,
            "--output-format", "text",
            full_prompt
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=180,  # 3 minute timeout for complex generation
                shell=False
            )
            
            if result.returncode != 0:
                error_msg = result.stderr.strip() if result.stderr else "Unknown error"
                raise RuntimeError(f"Claude Code CLI error: {error_msg}")
            
            output = result.stdout.strip()
            
            # Claude Code may include explanations, extract just the code
            if "```python" in output:
                output = output.split("```python")[1].split("```")[0]
            elif "```" in output:
                parts = output.split("```")
                if len(parts) >= 2:
                    output = parts[1]
            
            return output.strip()
            
        except subprocess.TimeoutExpired:
            raise RuntimeError("Claude Code timed out after 180 seconds")
        except FileNotFoundError:
            raise RuntimeError(
                "Claude Code CLI not found. Install it from: https://code.claude.com/docs/en/setup\n"
                "On Windows: irm https://claude.ai/install.ps1 | iex\n"
                "On macOS/Linux: curl -fsSL https://claude.ai/install.sh | bash"
            )

    def _generate_rule_based(self, func_info: Dict, code_str: str, user_instruction: str) -> str:
        """Generate Streamlit code using rule-based mapping."""
        func_name = func_info['name']
        title = func_name.replace('_', ' ').title()
        instruction_literal = json.dumps(user_instruction)
        primary_color = self._extract_primary_color(user_instruction)

        # Build sidebar inputs (no indentation - widgets are at top level in generated code)
        sidebar_code = []
        for param in func_info['parameters']:
            widget = self._generate_widget(param, user_instruction)
            sidebar_code.append(widget)

        # Build function call
        param_names = [p['name'] for p in func_info['parameters']]
        func_call = f"{func_name}({', '.join(param_names)})"

        # Detect domain from function name or code
        code_lower = code_str.lower()
        func_name_lower = func_name.lower()
        is_flood = 'flood' in func_name_lower or 'flood' in code_lower
        is_climate = 'climate' in func_name_lower or 'climate' in code_lower
        is_fire = 'fire' in func_name_lower or 'fwi' in code_lower
        is_geospatial = is_flood or is_climate or is_fire or 'lat' in code_lower

        # Determine output handling based on return type and user instruction
        wants_map = "map" in user_instruction.lower() or is_geospatial
        return_type = func_info['return_type']
        
        if 'Figure' in return_type or 'plt' in return_type:
            output_code = "st.pyplot(st.session_state.result)"
        elif 'Dict' in return_type and (wants_map or is_flood):
            # Dict with coordinates - show map and metrics (rich visualization)
            output_code = self._generate_rich_output_code(is_flood, is_climate, is_fire)
        elif 'Dict' in return_type:
            output_code = "st.json(st.session_state.result)"
        else:
            output_code = "st.write(st.session_state.result)"

        # Get docstring summary
        doc_summary = ""
        if func_info['docstring']:
            first_para = func_info['docstring'].split('\n\n')[0]
            doc_summary = ' '.join(first_para.split())

        css_block = ""
        if primary_color:
            css_block = f"""
<style>
:root {{
    --primary-color: {primary_color};
}}
.stButton>button {{
    background-color: {primary_color};
}}
</style>
"""

        # Generate the complete code
        # Safely escape user instruction for embedding
        safe_instruction = user_instruction.replace('"', '\\"')
        safe_doc_summary = doc_summary.replace('"', '\\"')
        
        # CSS block handling
        css_markdown = ""
        if css_block:
            css_markdown = f'\nst.markdown("""{css_block}""", unsafe_allow_html=True)'
        
        # Generate preview section for geospatial apps
        preview_section = ""
        if is_geospatial:
            preview_section = self._generate_preview_section(is_flood, is_climate, is_fire, func_info['parameters'])
        
        code = f'''"""
Auto-generated Streamlit UI for {func_name}
"""

import streamlit as st
import matplotlib.pyplot as plt

# Import the science function (paste the function code here or import from module)
# from science_module import {func_name}

# === PASTE YOUR SCIENCE FUNCTION HERE ===
{code_str}
# === END SCIENCE FUNCTION ===

# Page config
st.set_page_config(page_title="{title}", layout="wide")

# Title and description
st.title("{title}")
st.markdown("""{safe_doc_summary}""")
st.markdown("**User Instruction:** {safe_instruction}"){css_markdown}

# Initialize session state
if 'result' not in st.session_state:
    st.session_state.result = None

# Sidebar inputs
st.sidebar.header("Input Parameters")

{chr(10).join(sidebar_code)}

# Main area
st.divider()
{preview_section}
# Run button
if st.sidebar.button("Run Model", type="primary", use_container_width=True):
    with st.spinner("Running model..."):
        try:
            result = {func_call}
            st.session_state.result = result
            st.success("Model completed successfully!")
        except Exception as e:
            st.error(f"Error: {{str(e)}}")

# Display result
if st.session_state.result is not None:
    {output_code}
'''

        return code

    def _generate_widget(self, param: Dict, user_instruction: str) -> str:
        """Generate Streamlit widget code for a parameter."""
        name = param['name']
        ptype = param['type'].lower()
        default = param['default']
        desc = param['description'] or name.replace('_', ' ').title()
        label = name.replace('_', ' ').title()
        wants_slider = "slider" in user_instruction.lower()

        # Check for options (Literal type or docstring options)
        if param.get('options'):
            options = param['options']
            default_idx = options.index(default) if default in options else 0
            return f'{name} = st.sidebar.selectbox("{label}", options={options}, index={default_idx}, help="{desc}")'

        # Special handling for location/city selector (check if related functions exist)
        if ptype == 'str' and ('location' in name.lower() or 'city' in name.lower()):
            # Generate a selectbox with locations if get_flood_locations exists
            locations_list = '["Houston, TX", "Miami, FL", "New Orleans, LA", "Cedar Rapids, IA", "Sacramento, CA", "Charleston, SC", "Norfolk, VA", "Baton Rouge, LA"]'
            return f'{name} = st.sidebar.selectbox("{label}", options=get_flood_locations() if "get_flood_locations" in dir() else {locations_list}, help="{desc}")'

        # String with common options pattern
        if ptype == 'str':
            if default:
                return f'{name} = st.sidebar.text_input("{label}", value="{default}", help="{desc}")'
            return f'{name} = st.sidebar.text_input("{label}", help="{desc}")'

        # Integer - use number_input for large values, slider for small
        if ptype == 'int':
            val = default if default is not None else 50
            if val > 200:
                # Large integers: use number_input instead of slider
                return f'{name} = st.sidebar.number_input("{label}", min_value=1, value={val}, step=100, help="{desc}")'
            else:
                # Small integers: use slider with reasonable bounds
                max_val = max(100, val * 2) if val else 100
                return f'{name} = st.sidebar.slider("{label}", min_value=0, max_value={max_val}, value={val}, help="{desc}")'

        # Float
        if ptype == 'float':
            val = default if default is not None else 1.0
            if wants_slider:
                max_val = max(10.0, float(val) * 2) if val else 10.0
                return f'{name} = st.sidebar.slider("{label}", min_value=0.0, max_value={max_val}, value=float({val}), step=0.1, help="{desc}")'
            return f'{name} = st.sidebar.number_input("{label}", value={val}, step=0.1, help="{desc}")'

        # Boolean
        if ptype == 'bool':
            val = default if default is not None else False
            return f'{name} = st.sidebar.checkbox("{label}", value={val}, help="{desc}")'

        # Default: number input
        val = default if default is not None else 0
        return f'{name} = st.sidebar.number_input("{label}", value={val}, help="{desc}")'

    def _generate_preview_section(self, is_flood: bool, is_climate: bool, is_fire: bool, parameters: list) -> str:
        """Generate a preview section with map shown before Run button."""
        # Determine domain and emoji
        if is_flood:
            domain = "Flood"
            emoji = "🌊"
            locations_var = "FLOOD_LOCATIONS"
        elif is_climate:
            domain = "Climate"
            emoji = "🌡️"
            locations_var = "CLIMATE_LOCATIONS"
        elif is_fire:
            domain = "Fire"
            emoji = "🔥"
            locations_var = "FIRE_LOCATIONS"
        else:
            domain = "Analysis"
            emoji = "📊"
            locations_var = "LOCATIONS"
        
        # Find location parameter name
        location_param = "location_name"
        depth_param = None
        for p in parameters:
            if 'location' in p['name'].lower():
                location_param = p['name']
            if 'depth' in p['name'].lower() or 'param' in p['name'].lower():
                depth_param = p['name']
        
        return f'''
# Show preview before running (makes UI more attractive)
preview_col1, preview_col2 = st.columns([2, 1])

with preview_col1:
    st.subheader("{emoji} {domain} Risk Analysis")
    
    # Check package status
    try:
        status = check_{"flood" if is_flood else "climate" if is_climate else "fire" if is_fire else ""}_installation()
        if status.get('installed') and status.get('data_ready'):
            st.success(f"✅ Package active | Mode: **{{status.get('mode', 'direct')}}**")
        else:
            st.warning(f"⚠️ Simplified mode | Install: `{{status.get('install_command', 'pip install package')}}`")
    except:
        pass
    
    st.markdown(f"""
    **Selected Location:** {{{location_param}}}  
    {f"**Depth/Parameter:** {{{depth_param}:.1f}}" if depth_param else ""}
    
    Click **Run Model** to calculate with uncertainty quantification.
    """)

with preview_col2:
    st.subheader("🗺️ Location Preview")
    # Show selected location on map before running
    try:
        if {location_param} in {locations_var}:
            loc_info = {locations_var}[{location_param}]
            import pandas as pd
            preview_map = pd.DataFrame({{
                'lat': [loc_info.get('lat', 0)],
                'lon': [loc_info.get('lon', 0)],
            }})
            st.map(preview_map, zoom=8)
            st.caption(f"📍 {{{location_param}}}")
    except:
        st.info("Select a location to preview on map")

st.divider()
'''

    def _generate_rich_output_code(self, is_flood: bool, is_climate: bool, is_fire: bool) -> str:
        """Generate rich visualization code for geospatial/scientific results."""
        return '''result = st.session_state.result
    
    # Create two columns for results
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📊 Key Results")
        
        # Display metrics based on what's available
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        
        with metric_col1:
            if 'mean_loss' in result:
                st.metric("Mean Loss", f"${result['mean_loss']:,.0f}")
            elif 'mean' in result:
                st.metric("Mean", f"{result['mean']:.2f}")
            elif 'fwi' in result:
                st.metric("Fire Weather Index", f"{result['fwi']:.1f}")
        
        with metric_col2:
            if 'damage_ratio' in result:
                st.metric("Damage Ratio", f"{result['damage_ratio']:.1%}")
            elif 'std_loss' in result:
                st.metric("Std Dev", f"${result['std_loss']:,.0f}")
            elif 'std' in result:
                st.metric("Std Dev", f"{result['std']:.2f}")
        
        with metric_col3:
            if 'ci_lower' in result and 'ci_upper' in result:
                ci_text = f"${result['ci_lower']:,.0f} — ${result['ci_upper']:,.0f}"
                st.metric("95% CI", ci_text)
            elif 'using_unsafe' in result:
                mode = "✅ UNSAFE" if result['using_unsafe'] else "📊 Simplified"
                st.metric("Mode", mode)
        
        # Show confidence interval info
        if 'ci_lower' in result and 'ci_upper' in result:
            st.info(f"**95% Confidence Interval:** ${result.get('ci_lower', 0):,.0f} — ${result.get('ci_upper', 0):,.0f}")
        
        # Create histogram if losses available
        if 'losses' in result or 'mean_loss' in result:
            st.markdown("#### 📈 Loss Distribution")
            fig, ax = plt.subplots(figsize=(8, 4))
            if 'losses' in result:
                losses = result['losses']
            else:
                # Simulate distribution from mean and std
                import numpy as np
                mean = result.get('mean_loss', 10000)
                std = result.get('std_loss', mean * 0.3)
                losses = np.random.normal(mean, std, 1000)
                losses = np.maximum(losses, 0)
            ax.hist(losses, bins=30, edgecolor='white', alpha=0.7, color='steelblue')
            ax.axvline(result.get('mean_loss', np.mean(losses)), color='red', linestyle='--', 
                      linewidth=2, label='Mean')
            ax.set_xlabel('Loss ($)')
            ax.set_ylabel('Frequency')
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
    
    with col2:
        st.subheader("🗺️ Location")
        
        # Display map if coordinates available
        if 'lat' in result and 'lon' in result:
            import pandas as pd
            map_data = pd.DataFrame({
                'lat': [result['lat']],
                'lon': [result['lon']],
            })
            st.map(map_data, zoom=10)
            st.caption(f"📍 {result.get('location', 'Selected Location')}")
    
    # Details table
    st.divider()
    st.subheader("📋 Calculation Details")
    
    # Build details from result
    import pandas as pd
    details = []
    display_keys = ['location', 'flood_depth_m', 'property_value', 'foundation_type', 
                   'num_stories', 'num_simulations', 'using_unsafe', 'date', 'period']
    for key in display_keys:
        if key in result:
            value = result[key]
            if key == 'property_value':
                value = f"${value:,.0f}"
            elif key == 'flood_depth_m':
                value = f"{value:.2f} m"
            elif key == 'using_unsafe':
                value = "✅ Yes" if value else "❌ No"
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                value = f"{value:,}" if isinstance(value, int) else f"{value:.2f}"
            details.append({'Parameter': key.replace('_', ' ').title(), 'Value': str(value)})
    
    if details:
        details_df = pd.DataFrame(details)
        st.dataframe(details_df, hide_index=True, width="stretch")
    
    # Show full JSON in expander
    with st.expander("🔍 View Full Result JSON"):
        st.json(result)'''

    def _extract_primary_color(self, instruction: str) -> Optional[str]:
        """Pick a primary color based on keywords in the instruction."""
        if not instruction:
            return None
        text = instruction.lower()
        color_map = {
            "red": "#e53935",
            "blue": "#1e88e5",
            "green": "#2e7d32",
            "orange": "#fb8c00",
            "purple": "#8e24aa",
            "teal": "#00897b",
            "cyan": "#00acc1",
        }
        for word, color in color_map.items():
            if word in text:
                return color
        return None


# Convenience function for backward compatibility
def generate_ui_code(file_path: str, use_llm: bool = False, user_instruction: str = "") -> str:
    """
    Generate Streamlit UI code from a Python file.

    Args:
        file_path: Path to the Python file.
        use_llm: Whether to use LLM (requires API key in environment).
        user_instruction: Natural-language directions for layout/style.

    Returns:
        str: Generated Streamlit code.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        code_str = f.read()

    llm_client = None
    if use_llm:
        try:
            import openai
            llm_client = openai.OpenAI()
        except Exception:
            pass

    builder = StreamlitBuilder(llm_client)
    return builder.generate_ui_code(code_str, user_instruction)


if __name__ == "__main__":
    # Demo: Test with a sample code string
    sample_code = '''
def calculate_something(value: int, name: str = "test") -> plt.Figure:
    """
    Calculate something and return a plot.

    Args:
        value: The input value to process.
        name: Optional name for the calculation.

    Returns:
        plt.Figure: A matplotlib figure with the results.
    """
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [value, value*2, value*3])
    ax.set_title(f"Result: {name}")
    return fig
'''

    builder = StreamlitBuilder()

    print("=" * 60)
    print("Analyzing code...")
    print("=" * 60)
    info = builder.analyze_code(sample_code)
    print(f"Function: {info['name']}")
    print(f"Parameters: {info['parameters']}")
    print(f"Return type: {info['return_type']}")

    print("\n" + "=" * 60)
    print("Generated Streamlit code:")
    print("=" * 60)
    ui_code = builder.generate_ui_code(sample_code)
    print(ui_code)
