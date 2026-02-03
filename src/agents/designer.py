"""
TerraAgent v3.0: Designer Agent

🎨 The Designer Agent
Task: Based on User Natural Language Requirements and the "Verified Code Snippets" 
from the Engineer Agent, it writes streamlit_app.py.
Capability: It does not generate code blindly; it only calls functions that have 
been verified to run.
"""

import os
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional


@dataclass
class GeneratedApp:
    """Result of app generation."""
    code: str
    filename: str
    imports_added: List[str]
    ui_elements: List[str]


class DesignerAgent:
    """
    Generates Streamlit applications from verified code snippets.
    
    This agent creates UI code that properly imports and calls functions
    that have been verified to work by the Engineer Agent.
    """
    
    SYSTEM_PROMPT = '''You are a senior Streamlit Developer. Generate a complete Streamlit application based on the provided function signature and user instructions.

Code Structure Rules:

1. Imports: Import streamlit as st, matplotlib.pyplot as plt, and necessary scientific libraries.

2. sys.path: MUST add the repo path to sys.path at the top:
   ```python
   import sys
   sys.path.insert(0, "{repo_path}")
   ```

3. State Management: Use st.session_state to prevent re-running the model on every interaction.

4. Layout: Use st.sidebar for all input parameters:
   - int -> st.slider with appropriate range
   - float -> st.slider with step=0.1
   - str with options -> st.selectbox

5. Action: Include a st.button('Run Model'). Only call the science function when clicked.

6. Output Handling:
   - If return type is plt.Figure, use st.pyplot(figure)
   - If return type is pd.DataFrame, use st.dataframe()
   - Display relevant statistics and metadata
   - If the result contains lat/lon, show st.map() plus st.metric() for key values

7. UX: If the app is geospatial, show a preview section (summary + map) BEFORE the Run button.

8. Styling: Follow the user's styling instructions for colors, layout, etc.

Output ONLY valid Python code. No explanations or markdown.'''

    def __init__(self, llm_client: Optional[Any] = None, log_callback: Optional[Callable[[str], None]] = None):
        """
        Initialize the Designer Agent.
        
        Args:
            llm_client: Optional LLM client for enhanced generation.
            log_callback: Optional callback function for logging.
        """
        self.llm_client = llm_client
        self.log = log_callback or (lambda msg: print(f"[Designer] {msg}"))
        
    def _get_input_widget(self, param: Dict[str, Any], user_instruction: str = "") -> str:
        """Generate appropriate Streamlit input widget for a parameter."""
        name = param.get("name", "input")
        p_type = param.get("type", "Any").lower()
        default = param.get("default")
        
        # Parse user instruction for specific requirements
        instruction_lower = user_instruction.lower()
        
        if "int" in p_type:
            # Try to infer range from parameter name
            if "year" in name.lower():
                min_val, max_val, default_val = 2024, 2100, 2050
            elif "simulation" in name.lower() or "num" in name.lower():
                min_val, max_val, default_val = 100, 10000, 5000
            else:
                min_val, max_val, default_val = 0, 100, 50
            
            return f'st.sidebar.slider("{name.replace("_", " ").title()}", {min_val}, {max_val}, {default_val})'
            
        elif "float" in p_type:
            if "temperature" in name.lower() or "temp" in name.lower():
                min_val, max_val, default_val = -20.0, 50.0, 25.0
            elif "depth" in name.lower():
                min_val, max_val, default_val = 0.0, 10.0, 1.5
            elif "speed" in name.lower():
                min_val, max_val, default_val = 0.0, 100.0, 20.0
            elif "rain" in name.lower():
                min_val, max_val, default_val = 0.0, 100.0, 0.0
            else:
                min_val, max_val, default_val = 0.0, 100.0, 50.0
            
            return f'st.sidebar.slider("{name.replace("_", " ").title()}", {min_val}, {max_val}, {default_val}, step=0.1)'
            
        elif "str" in p_type:
            # Check for common string parameters
            if "scenario" in name.lower() or "emission" in name.lower():
                options = '["Low", "Medium", "High"]'
            elif "mode" in name.lower():
                options = '["default", "advanced"]'
            else:
                options = '["Option A", "Option B", "Option C"]'
            
            return f'st.sidebar.selectbox("{name.replace("_", " ").title()}", {options})'
            
        else:
            # Default to text input
            return f'st.sidebar.text_input("{name.replace("_", " ").title()}")'
    
    def _generate_rule_based(
        self,
        function_info: Dict[str, Any],
        repo_path: str,
        user_instruction: str
    ) -> str:
        """Generate Streamlit code using rule-based approach."""
        func_name = function_info.get("name", "main")
        file_path = function_info.get("file_path", "main.py")
        params = function_info.get("parameters", [])
        return_type = function_info.get("return_type", "Any")
        docstring = function_info.get("docstring", "")
        
        # Generate module import path
        module_path = file_path.replace("/", ".").replace("\\", ".").replace(".py", "")
        
        # Generate input widgets
        input_widgets = []
        param_vars = []
        for p in params:
            widget = self._get_input_widget(p, user_instruction)
            var_name = p.get("name", "input")
            input_widgets.append(f'{var_name} = {widget}')
            param_vars.append(var_name)
        
        # Generate output handling
        if "Figure" in str(return_type) or "plt" in str(return_type):
            output_code = '''
    with st.spinner("Running model..."):
        result = {func_name}({params})
        st.pyplot(result)
        st.success("Model executed successfully!")
'''.format(func_name=func_name, params=", ".join(param_vars))
        elif "DataFrame" in str(return_type):
            output_code = '''
    with st.spinner("Running model..."):
        result = {func_name}({params})
        st.dataframe(result)
        st.success("Model executed successfully!")
'''.format(func_name=func_name, params=", ".join(param_vars))
        else:
            output_code = '''
    with st.spinner("Running model..."):
        result = {func_name}({params})
        st.write("Result:", result)
        st.success("Model executed successfully!")
'''.format(func_name=func_name, params=", ".join(param_vars))
        
        # Parse user instruction for title
        title = f"🌍 {func_name.replace('_', ' ').title()}"
        if "climate" in user_instruction.lower():
            title = "🌤️ Climate Simulation"
        elif "fire" in user_instruction.lower():
            title = "🔥 Fire Risk Assessment"
        elif "flood" in user_instruction.lower():
            title = "🌊 Flood Loss Analysis"
        
        # Build the complete app
        code = f'''"""
Generated Streamlit App by TerraAgent Designer Agent
Function: {func_name}
Source: {file_path}
"""

import sys
sys.path.insert(0, r"{repo_path}")

import streamlit as st
import matplotlib.pyplot as plt

# Import the science function
from {module_path} import {func_name}

# Page configuration
st.set_page_config(
    page_title="{title}",
    page_icon="🌍",
    layout="wide"
)

# Title and description
st.title("{title}")
st.markdown("""
{docstring[:200] if docstring else f"Interactive interface for {func_name}"}
""")

# Sidebar inputs
st.sidebar.header("Model Parameters")
{chr(10).join(input_widgets)}

# Initialize session state
if "result" not in st.session_state:
    st.session_state.result = None

# Run button
if st.sidebar.button("🚀 Run Model", type="primary", use_container_width=True):
{output_code}

# Display previous result if available
elif st.session_state.result is not None:
    st.info("Showing previous result. Adjust parameters and click Run to update.")
'''
        
        return code
    
    def _generate_with_llm(
        self,
        function_info: Dict[str, Any],
        repo_path: str,
        user_instruction: str,
        working_snippet: Optional[str] = None
    ) -> str:
        """Generate Streamlit code using LLM."""
        prompt = f"""Generate a Streamlit application for this function:

Function: {function_info.get('name')}
File: {function_info.get('file_path')}
Signature: {function_info.get('signature', 'unknown')}
Docstring: {function_info.get('docstring', 'N/A')}
Return Type: {function_info.get('return_type', 'Any')}
Parameters: {function_info.get('parameters', [])}

Repository Path (MUST add to sys.path): {repo_path}

User Instructions: {user_instruction}

Working Code Snippet (verified to run):
```python
{working_snippet or 'N/A'}
```

Generate a complete, working Streamlit application following all the rules in the system prompt.
"""
        
        system_prompt = self.SYSTEM_PROMPT.format(repo_path=repo_path)

        try:
            code = self._call_llm(system_prompt, prompt)

            # Clean up code (remove markdown fences if present)
            code = re.sub(r'^```python\n?', '', code)
            code = re.sub(r'\n?```$', '', code)

            return code.strip()

        except Exception as e:
            self.log(f"LLM generation failed: {e}, falling back to rule-based")
            return self._generate_rule_based(function_info, repo_path, user_instruction)

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Call the LLM based on provider/client type."""
        # Dict-based client (new multi-provider format)
        if isinstance(self.llm_client, dict):
            client = self.llm_client.get("client")
            model = self.llm_client.get("model")
            provider = self.llm_client.get("provider")

            if provider == "claude_code":
                return self._call_claude_code(system_prompt, user_prompt, model)

            if provider in {"openai", "openrouter", "ollama"}:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.3
                )
                return response.choices[0].message.content

            if provider == "anthropic":
                response = client.messages.create(
                    model=model,
                    max_tokens=4096,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}]
                )
                return response.content[0].text

            if provider == "google":
                model_instance = client.GenerativeModel(model)
                full_prompt = f"{system_prompt}\n\n{user_prompt}"
                response = model_instance.generate_content(full_prompt)
                return response.text

            if provider == "mistral":
                response = client.chat.complete(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ]
                )
                return response.choices[0].message.content

            raise ValueError(f"Unsupported LLM provider: {provider}")

        # Legacy: OpenAI-style client
        if hasattr(self.llm_client, 'chat'):
            response = self.llm_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3
            )
            return response.choices[0].message.content

        # Legacy: Anthropic-style client
        if hasattr(self.llm_client, 'messages'):
            response = self.llm_client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=4096,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}]
            )
            return response.content[0].text

        raise ValueError("Unsupported LLM client type")

    def _call_claude_code(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str = "sonnet"
    ) -> str:
        """Call Claude Code CLI for generation."""
        import subprocess

        full_prompt = f"""{system_prompt}

---

{user_prompt}"""

        cmd = [
            "claude",
            "-p",
            "--model", model,
            "--output-format", "text",
            "--dangerously-skip-permissions",
            full_prompt
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=180
        )

        if result.returncode != 0:
            error_msg = result.stderr.strip() if result.stderr else "Unknown error"
            raise RuntimeError(f"Claude Code CLI error: {error_msg}")

        return result.stdout.strip()
    
    def design(
        self,
        function_info: Dict[str, Any],
        repo_path: str,
        user_instruction: str,
        working_snippet: Optional[str] = None
    ) -> GeneratedApp:
        """
        Main entry point: Generate a Streamlit application.
        
        Args:
            function_info: Dictionary with function details.
            repo_path: Path to the repository.
            user_instruction: Natural language requirements from user.
            working_snippet: Verified working code from Engineer Agent.
            
        Returns:
            GeneratedApp with the generated code.
        """
        self.log("=" * 50)
        self.log("Starting UI generation")
        self.log("=" * 50)
        
        func_name = function_info.get("name", "main")
        self.log(f"Generating UI for function: {func_name}")
        self.log(f"User instruction: {user_instruction[:100]}...")
        
        if self.llm_client:
            self.log("Using LLM for enhanced generation")
            code = self._generate_with_llm(
                function_info,
                repo_path,
                user_instruction,
                working_snippet
            )
        else:
            self.log("Using rule-based generation")
            code = self._generate_rule_based(
                function_info,
                repo_path,
                user_instruction
            )
        
        self.log(f"✓ Generated {len(code)} characters of code")
        
        # Parse generated code for metadata
        imports = re.findall(r'^(?:from|import)\s+\S+', code, re.MULTILINE)
        ui_elements = re.findall(r'st\.(?:sidebar\.)?(\w+)', code)
        
        return GeneratedApp(
            code=code,
            filename="generated_app.py",
            imports_added=imports,
            ui_elements=list(set(ui_elements))
        )
    
    def save_app(self, app: GeneratedApp, output_path: str) -> str:
        """
        Save the generated app to a file.
        
        Args:
            app: GeneratedApp object.
            output_path: Directory to save the file.
            
        Returns:
            Full path to the saved file.
        """
        filepath = os.path.join(output_path, app.filename)
        
        with open(filepath, 'w') as f:
            f.write(app.code)
            
        self.log(f"✓ App saved to {filepath}")
        return filepath


if __name__ == "__main__":
    # Test
    agent = DesignerAgent()
    print("Designer Agent initialized. Use design(function_info, repo_path, instruction) to generate UI.")
