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
    SYSTEM_PROMPT = """You are a senior Streamlit Developer. I will give you a Python function signature. Please write a complete Streamlit application for it.

Code Structure Rules:

Imports: Import streamlit as st, matplotlib.pyplot as plt, and the necessary scientific libraries.

State Management: Use st.session_state to prevent re-running the model on every interaction.

Layout: Use st.sidebar for all input parameters (int -> sliders, str -> selectbox). Use the main area for the Output.

Action: Crucial! Include a st.button('Run Model'). Only call the science function when clicked.

Output Handling: Detect the return type. If plt.Figure, use st.pyplot(). If pd.DataFrame, use st.dataframe().

Output ONLY the Python code, no explanations or markdown."""

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

        # Find the first public function
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and not node.name.startswith('_'):
                return self._extract_function_info(node, code_str)

        return {}

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
        """Generate Streamlit code using LLM."""
        # Build the prompt
        params_str = ', '.join([
            f"{p['name']}: {p['type']}" + (f" = {p['default']}" if p['default'] is not None else "")
            for p in func_info['parameters']
        ])
        signature = f"def {func_info['name']}({params_str}) -> {func_info['return_type']}"

        user_prompt = f"""Generate a Streamlit app for this function:

```python
{signature}
```

Docstring:
{func_info['docstring']}

User Instruction:
\"\"\"{user_instruction}\"\"\"

Full source code:
```python
{code_str}
```"""

        try:
            # Try OpenAI-style API
            if hasattr(self.llm_client, 'chat'):
                response = self.llm_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": self.SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0
                )
                code = response.choices[0].message.content
            else:
                # Try Anthropic-style API
                response = self.llm_client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=4096,
                    system=self.SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": user_prompt}]
                )
                code = response.content[0].text

            # Extract code from markdown if present
            if "```python" in code:
                code = code.split("```python")[1].split("```")[0]
            elif "```" in code:
                code = code.split("```")[1].split("```")[0]

            return code.strip()

        except Exception as e:
            print(f"LLM generation failed: {e}, falling back to rule-based")
            return self._generate_rule_based(func_info, code_str, user_instruction)

    def _generate_rule_based(self, func_info: Dict, code_str: str, user_instruction: str) -> str:
        """Generate Streamlit code using rule-based mapping."""
        func_name = func_info['name']
        title = func_name.replace('_', ' ').title()
        instruction_literal = json.dumps(user_instruction)
        primary_color = self._extract_primary_color(user_instruction)

        # Build sidebar inputs
        sidebar_code = []
        for param in func_info['parameters']:
            widget = self._generate_widget(param, user_instruction)
            sidebar_code.append(f"    {widget}")

        # Build function call
        param_names = [p['name'] for p in func_info['parameters']]
        func_call = f"{func_name}({', '.join(param_names)})"

        # Determine output handling
        if 'Figure' in func_info['return_type'] or 'plt' in func_info['return_type']:
            output_code = "st.pyplot(result)"
        else:
            output_code = "st.write(result)"

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
        code = f'''"""
Auto-generated Streamlit UI for {func_name}
"""

import streamlit as st
import matplotlib.pyplot as plt
import json

# Import the science function (paste the function code here or import from module)
# from science_module import {func_name}

# === PASTE YOUR SCIENCE FUNCTION HERE ===
{code_str}
# === END SCIENCE FUNCTION ===

# Page config
st.set_page_config(page_title="{title}", layout="wide")

# Title and description
st.title("{title}")
st.markdown("""{doc_summary}""")
st.markdown("**User Instruction:** " + json.dumps({instruction_literal}))
st.markdown({json.dumps(css_block)}, unsafe_allow_html=True)

# Initialize session state
if 'result' not in st.session_state:
    st.session_state.result = None

# Sidebar inputs
st.sidebar.header("Input Parameters")

{chr(10).join(sidebar_code)}

# Main area
st.divider()

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

        # String with common options pattern
        if ptype == 'str':
            if default:
                return f'{name} = st.sidebar.text_input("{label}", value="{default}", help="{desc}")'
            return f'{name} = st.sidebar.text_input("{label}", help="{desc}")'

        # Integer
        if ptype == 'int':
            val = default if default is not None else 50
            return f'{name} = st.sidebar.slider("{label}", min_value=0, max_value=200, value={val}, help="{desc}")'

        # Float
        if ptype == 'float':
            val = default if default is not None else 1.0
            if wants_slider:
                return f'{name} = st.sidebar.slider("{label}", min_value=0.0, max_value=10.0, value=float({val}), step=0.1, help="{desc}")'
            return f'{name} = st.sidebar.number_input("{label}", value={val}, step=0.1, help="{desc}")'

        # Boolean
        if ptype == 'bool':
            val = default if default is not None else False
            return f'{name} = st.sidebar.checkbox("{label}", value={val}, help="{desc}")'

        # Default: number input
        val = default if default is not None else 0
        return f'{name} = st.sidebar.number_input("{label}", value={val}, help="{desc}")'

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
