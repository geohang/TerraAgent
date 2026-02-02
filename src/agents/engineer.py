"""
TerraAgent v3.0: Engineer Agent

🔧 The Engineer Agent
Task: Parse requirements.txt or pyproject.toml.
Core Capability: Sandbox Execution - attempts to pip install in an isolated 
environment and writes a minimal test_script.py to verify if the code runs.
Self-Correction: If an error occurs, reads the error log and attempts to 
automatically fix the environment.
"""

import os
import subprocess
import sys
import tempfile
import venv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


@dataclass
class VerificationResult:
    """Result of code verification."""
    success: bool
    working_snippet: Optional[str]
    error_log: Optional[str]
    imported_modules: List[str]


class EngineerAgent:
    """
    Sets up virtual environments and verifies code execution.
    
    This agent creates sandboxed environments, installs dependencies,
    and runs smoke tests to verify that imported code actually works.
    """
    
    def __init__(self, log_callback: Optional[Callable[[str], None]] = None):
        """
        Initialize the Engineer Agent.
        
        Args:
            log_callback: Optional callback function for logging messages.
        """
        self.log = log_callback or (lambda msg: print(f"[Engineer] {msg}"))
        
    def setup_venv(self, repo_path: str) -> str:
        """
        Create a virtual environment in the repository directory.
        
        Args:
            repo_path: Path to the repository.
            
        Returns:
            Path to the virtual environment.
        """
        venv_path = os.path.join(repo_path, ".venv")
        
        self.log(f"Creating virtual environment at {venv_path}...")
        
        try:
            # Create venv with pip
            venv.create(venv_path, with_pip=True)
            self.log("✓ Virtual environment created successfully")
            return venv_path
        except Exception as e:
            self.log(f"✗ Failed to create venv: {e}")
            raise RuntimeError(f"Virtual environment creation failed: {e}")
    
    def get_python_executable(self, venv_path: str) -> str:
        """Get the Python executable path for the venv."""
        if sys.platform == "win32":
            return os.path.join(venv_path, "Scripts", "python.exe")
        else:
            return os.path.join(venv_path, "bin", "python")
    
    def get_pip_executable(self, venv_path: str) -> str:
        """Get the pip executable path for the venv."""
        if sys.platform == "win32":
            return os.path.join(venv_path, "Scripts", "pip.exe")
        else:
            return os.path.join(venv_path, "bin", "pip")
    
    def parse_requirements(self, repo_path: str) -> List[str]:
        """
        Parse requirements from requirements.txt or pyproject.toml.
        
        Args:
            repo_path: Path to the repository.
            
        Returns:
            List of package requirements.
        """
        requirements = []
        
        # Try requirements.txt
        req_path = os.path.join(repo_path, "requirements.txt")
        if os.path.exists(req_path):
            self.log("Parsing requirements.txt...")
            with open(req_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and not line.startswith('-'):
                        requirements.append(line)
            self.log(f"✓ Found {len(requirements)} packages in requirements.txt")
            return requirements
            
        # Try pyproject.toml
        pyproject_path = os.path.join(repo_path, "pyproject.toml")
        if os.path.exists(pyproject_path):
            self.log("Parsing pyproject.toml...")
            try:
                import tomllib
            except ImportError:
                import tomli as tomllib
                
            with open(pyproject_path, 'rb') as f:
                data = tomllib.load(f)
                
            # Get dependencies from different possible locations
            deps = (
                data.get("project", {}).get("dependencies", []) or
                data.get("tool", {}).get("poetry", {}).get("dependencies", {})
            )
            
            if isinstance(deps, dict):
                # Poetry style
                requirements = [k for k in deps.keys() if k != "python"]
            else:
                requirements = list(deps)
                
            self.log(f"✓ Found {len(requirements)} packages in pyproject.toml")
            return requirements
            
        self.log("  No dependency file found")
        return []
    
    def install_deps(self, repo_path: str, venv_path: str) -> Tuple[bool, str]:
        """
        Install dependencies in the virtual environment.
        
        Args:
            repo_path: Path to the repository.
            venv_path: Path to the virtual environment.
            
        Returns:
            Tuple of (success, error_log or success message).
        """
        pip = self.get_pip_executable(venv_path)
        
        # First, upgrade pip
        self.log("Upgrading pip...")
        try:
            subprocess.run(
                [pip, "install", "--upgrade", "pip"],
                capture_output=True,
                text=True,
                timeout=120
            )
        except Exception as e:
            self.log(f"  Warning: pip upgrade failed: {e}")
        
        requirements = self.parse_requirements(repo_path)
        
        if not requirements:
            self.log("  No dependencies to install")
            return True, "No dependencies required"
        
        self.log(f"Installing {len(requirements)} packages...")
        
        # Try to install requirements.txt directly if it exists
        req_path = os.path.join(repo_path, "requirements.txt")
        if os.path.exists(req_path):
            try:
                result = subprocess.run(
                    [pip, "install", "-r", req_path],
                    capture_output=True,
                    text=True,
                    timeout=600
                )
                
                if result.returncode == 0:
                    self.log("✓ All dependencies installed successfully")
                    return True, "Dependencies installed successfully"
                else:
                    error_msg = result.stderr or result.stdout
                    self.log(f"✗ Installation failed: {error_msg[:200]}")
                    return False, error_msg
                    
            except subprocess.TimeoutExpired:
                self.log("✗ Installation timed out")
                return False, "Installation timed out after 10 minutes"
            except Exception as e:
                self.log(f"✗ Installation error: {e}")
                return False, str(e)
        
        # Install packages one by one if no requirements.txt
        failed = []
        for pkg in requirements:
            try:
                result = subprocess.run(
                    [pip, "install", pkg],
                    capture_output=True,
                    text=True,
                    timeout=120
                )
                if result.returncode != 0:
                    failed.append((pkg, result.stderr))
            except Exception as e:
                failed.append((pkg, str(e)))
                
        if failed:
            error_log = "\n".join([f"{pkg}: {err}" for pkg, err in failed])
            self.log(f"✗ {len(failed)} packages failed to install")
            return False, error_log
            
        self.log("✓ All dependencies installed successfully")
        return True, "Dependencies installed successfully"
    
    def verify_import(self, venv_path: str, module_name: str, repo_path: str) -> Tuple[bool, str]:
        """
        Try to import a module in the virtual environment.
        
        Args:
            venv_path: Path to the virtual environment.
            module_name: Name of the module to import.
            repo_path: Path to add to sys.path.
            
        Returns:
            Tuple of (success, error_message or success confirmation).
        """
        python = self.get_python_executable(venv_path)
        
        # Create import test script
        test_script = f'''
import sys
sys.path.insert(0, r"{repo_path}")
try:
    import {module_name}
    print("SUCCESS: Module imported successfully")
except Exception as e:
    print(f"ERROR: {{e}}")
    sys.exit(1)
'''
        
        try:
            result = subprocess.run(
                [python, "-c", test_script],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                return True, f"Module '{module_name}' imported successfully"
            else:
                error = result.stderr or result.stdout
                return False, error
                
        except subprocess.TimeoutExpired:
            return False, "Import timed out"
        except Exception as e:
            return False, str(e)
    
    def write_smoke_test(self, repo_path: str, function_info: Dict[str, Any]) -> str:
        """
        Generate a minimal smoke test script for a function.
        
        Args:
            repo_path: Path to the repository.
            function_info: Dictionary with function details.
            
        Returns:
            Path to the generated test script.
        """
        func_name = function_info.get("name", "main")
        file_path = function_info.get("file_path", "main.py")
        params = function_info.get("parameters", [])
        
        # Generate module path from file path
        module_path = file_path.replace("/", ".").replace("\\", ".").replace(".py", "")
        
        # Generate default arguments
        default_args = []
        for p in params:
            p_type = p.get("type", "Any")
            if "int" in p_type.lower():
                default_args.append("2050")
            elif "float" in p_type.lower():
                default_args.append("25.0")
            elif "str" in p_type.lower():
                default_args.append('"Medium"')
            else:
                default_args.append("None")
        
        args_str = ", ".join(default_args)
        
        test_script = f'''#!/usr/bin/env python3
"""Smoke test generated by TerraAgent Engineer Agent"""
import sys
sys.path.insert(0, r"{repo_path}")

try:
    from {module_path} import {func_name}
    print(f"SUCCESS: Imported {func_name} from {module_path}")
    
    # Try to run the function with default arguments
    result = {func_name}({args_str})
    print(f"SUCCESS: Function executed, returned type: {{type(result).__name__}}")
    
except Exception as e:
    print(f"ERROR: {{e}}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
'''
        
        test_path = os.path.join(repo_path, "test_script.py")
        with open(test_path, 'w') as f:
            f.write(test_script)
            
        self.log(f"✓ Generated smoke test: {test_path}")
        return test_path
    
    def run_smoke_test(self, venv_path: str, test_script_path: str) -> VerificationResult:
        """
        Run the smoke test script.
        
        Args:
            venv_path: Path to the virtual environment.
            test_script_path: Path to the test script.
            
        Returns:
            VerificationResult with test outcome.
        """
        python = self.get_python_executable(venv_path)
        self.log("Running smoke test...")
        
        try:
            result = subprocess.run(
                [python, test_script_path],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            output = result.stdout + result.stderr
            
            if result.returncode == 0 and "SUCCESS" in output:
                self.log("✓ Smoke test passed!")
                
                # Read the test script as the working snippet
                with open(test_script_path, 'r') as f:
                    working_snippet = f.read()
                    
                return VerificationResult(
                    success=True,
                    working_snippet=working_snippet,
                    error_log=None,
                    imported_modules=[]
                )
            else:
                self.log(f"✗ Smoke test failed: {output[:200]}")
                return VerificationResult(
                    success=False,
                    working_snippet=None,
                    error_log=output,
                    imported_modules=[]
                )
                
        except subprocess.TimeoutExpired:
            self.log("✗ Smoke test timed out")
            return VerificationResult(
                success=False,
                working_snippet=None,
                error_log="Test timed out after 60 seconds",
                imported_modules=[]
            )
        except Exception as e:
            self.log(f"✗ Smoke test error: {e}")
            return VerificationResult(
                success=False,
                working_snippet=None,
                error_log=str(e),
                imported_modules=[]
            )
    
    def attempt_fix(self, error_log: str, venv_path: str) -> bool:
        """
        Attempt to fix environment issues based on error log.
        
        Args:
            error_log: The error message from failed installation/import.
            venv_path: Path to the virtual environment.
            
        Returns:
            True if a fix was attempted, False otherwise.
        """
        pip = self.get_pip_executable(venv_path)
        
        # Common fixes based on error patterns
        fixes = []
        
        if "ModuleNotFoundError" in error_log:
            # Extract missing module name
            import re
            match = re.search(r"No module named '([^']+)'", error_log)
            if match:
                module = match.group(1).split('.')[0]
                fixes.append(module)
                
        if "numpy" in error_log.lower() and "numpy" not in fixes:
            fixes.append("numpy")
            
        if "matplotlib" in error_log.lower() and "matplotlib" not in fixes:
            fixes.append("matplotlib")
            
        if not fixes:
            return False
            
        self.log(f"Attempting to fix by installing: {fixes}")
        
        for pkg in fixes:
            try:
                subprocess.run(
                    [pip, "install", pkg],
                    capture_output=True,
                    timeout=120
                )
            except:
                pass
                
        return True
    
    def verify(
        self,
        repo_path: str,
        function_info: Optional[Dict[str, Any]] = None,
        max_retries: int = 2
    ) -> VerificationResult:
        """
        Main entry point: Set up environment and verify code execution.
        
        Args:
            repo_path: Path to the repository.
            function_info: Optional function to test (uses main if not provided).
            max_retries: Maximum number of fix attempts.
            
        Returns:
            VerificationResult with verification outcome.
        """
        self.log("=" * 50)
        self.log("Starting environment verification")
        self.log("=" * 50)
        
        # Setup venv
        venv_path = self.setup_venv(repo_path)
        
        # Install dependencies
        success, error = self.install_deps(repo_path, venv_path)
        
        if not success:
            # Try to fix and retry
            for i in range(max_retries):
                self.log(f"Retry {i+1}/{max_retries}: Attempting to fix...")
                if self.attempt_fix(error, venv_path):
                    success, error = self.install_deps(repo_path, venv_path)
                    if success:
                        break
        
        if not success:
            return VerificationResult(
                success=False,
                working_snippet=None,
                error_log=error,
                imported_modules=[]
            )
        
        # If we have function info, run a smoke test
        if function_info:
            test_path = self.write_smoke_test(repo_path, function_info)
            result = self.run_smoke_test(venv_path, test_path)
            
            # Retry on failure
            if not result.success and result.error_log:
                for i in range(max_retries):
                    self.log(f"Retry {i+1}/{max_retries}: Attempting to fix test failure...")
                    if self.attempt_fix(result.error_log, venv_path):
                        result = self.run_smoke_test(venv_path, test_path)
                        if result.success:
                            break
                            
            return result
        
        # No function info, just verify environment is set up
        return VerificationResult(
            success=True,
            working_snippet="# Environment verified successfully",
            error_log=None,
            imported_modules=[]
        )

    def install_github_package(
        self,
        github_url: str,
        extra_deps: Optional[List[str]] = None,
        use_current_env: bool = True
    ) -> Tuple[bool, str, List[str]]:
        """
        Install a package directly from a GitHub repository.
        
        This is the preferred method for integrating external scientific packages.
        Uses pip install git+{url} syntax.
        
        Args:
            github_url: GitHub repository URL (e.g., https://github.com/owner/repo).
            extra_deps: Additional dependencies to install (e.g., ['geopandas', 'rasterio']).
            use_current_env: If True, install in current environment (not a venv).
            
        Returns:
            Tuple of (success, message, list of verified imports).
            
        Example:
            >>> result = engineer.install_github_package(
            ...     "https://github.com/abpoll/unsafe",
            ...     extra_deps=["geopandas", "pyyaml"]
            ... )
            >>> print(result)
            (True, "Package installed successfully", ["unsafe.ddfs", "unsafe.ensemble"])
        """
        self.log(f"Installing package from GitHub: {github_url}")
        
        # Determine pip executable
        if use_current_env:
            pip = sys.executable.replace("python", "pip")
            if not os.path.exists(pip):
                pip = [sys.executable, "-m", "pip"]
            else:
                pip = [pip]
        else:
            # Would need venv_path parameter
            raise ValueError("Non-current environment installation requires venv_path")
        
        # Install the package from GitHub
        install_url = f"git+{github_url}"
        self.log(f"Running: pip install {install_url}")
        
        try:
            result = subprocess.run(
                pip + ["install", install_url],
                capture_output=True,
                text=True,
                timeout=600
            )
            
            if result.returncode != 0:
                error_msg = result.stderr or result.stdout
                self.log(f"✗ Installation failed: {error_msg[:300]}")
                return False, error_msg, []
                
            self.log("✓ Package installed successfully")
            
        except subprocess.TimeoutExpired:
            self.log("✗ Installation timed out")
            return False, "Installation timed out after 10 minutes", []
        except Exception as e:
            self.log(f"✗ Installation error: {e}")
            return False, str(e), []
        
        # Install extra dependencies
        if extra_deps:
            self.log(f"Installing extra dependencies: {extra_deps}")
            for dep in extra_deps:
                try:
                    subprocess.run(
                        pip + ["install", dep],
                        capture_output=True,
                        text=True,
                        timeout=120
                    )
                except Exception as e:
                    self.log(f"  Warning: Failed to install {dep}: {e}")
        
        # Extract package name from URL
        package_name = github_url.rstrip('/').split('/')[-1]
        
        # Try to verify import
        verified_imports = []
        try:
            # Try importing the main package
            test_code = f"import {package_name}; print('OK')"
            result = subprocess.run(
                [sys.executable, "-c", test_code],
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode == 0:
                verified_imports.append(package_name)
                self.log(f"✓ Verified import: {package_name}")
        except Exception:
            pass
        
        return True, "Package installed successfully", verified_imports
    
    def download_data_file(
        self,
        url: str,
        output_path: str,
        source_type: str = "auto"
    ) -> Tuple[bool, str]:
        """
        Download a data file from a URL.
        
        Supports Zenodo, GitHub raw files, and direct URLs.
        
        Args:
            url: URL to download from.
            output_path: Local path to save the file.
            source_type: Type of source ("zenodo", "github", or "auto").
            
        Returns:
            Tuple of (success, message or error).
        """
        import urllib.request
        from pathlib import Path
        
        self.log(f"Downloading data from: {url}")
        
        # Create output directory if needed
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        try:
            urllib.request.urlretrieve(url, output_path)
            self.log(f"✓ Downloaded to: {output_path}")
            return True, f"Downloaded to {output_path}"
        except Exception as e:
            self.log(f"✗ Download failed: {e}")
            return False, str(e)
    
    def verify_package_import(
        self,
        package_name: str,
        submodules: Optional[List[str]] = None
    ) -> Dict[str, bool]:
        """
        Verify that a package and its submodules can be imported.
        
        Args:
            package_name: Main package name.
            submodules: List of submodule names to verify.
            
        Returns:
            Dict mapping module names to import success status.
        """
        results = {}
        
        modules_to_check = [package_name]
        if submodules:
            modules_to_check.extend([f"{package_name}.{sub}" for sub in submodules])
        
        for module in modules_to_check:
            try:
                test_code = f"import {module}; print('OK')"
                result = subprocess.run(
                    [sys.executable, "-c", test_code],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                results[module] = (result.returncode == 0)
            except Exception:
                results[module] = False
        
        return results


if __name__ == "__main__":
    # Test
    agent = EngineerAgent()
    print("Engineer Agent initialized. Use verify(repo_path, function_info) to test code.")
