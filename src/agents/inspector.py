"""
TerraAgent v3.0: Inspector Agent

🕵️‍♂️ The Inspector Agent
Task: git clone the repository, traverse the file tree, and identify main.py, 
core classes, and data structures.
Capability: If the code is messy, it uses an LLM to summarize the "Entry Point".
"""

import ast
import json
import os
import shutil
import tempfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable
import git


@dataclass
class FunctionInfo:
    """Information about a function in the repository."""
    name: str
    file_path: str
    line_number: int
    signature: str
    docstring: Optional[str]
    return_type: Optional[str]
    parameters: List[Dict[str, str]]


@dataclass
class ClassInfo:
    """Information about a class in the repository."""
    name: str
    file_path: str
    line_number: int
    docstring: Optional[str]
    methods: List[str]
    bases: List[str]


@dataclass
class RepoSummary:
    """Summary of a cloned repository."""
    repo_path: str
    entry_points: List[str]
    python_files: List[str]
    functions: List[FunctionInfo]
    classes: List[ClassInfo]
    requirements_file: Optional[str]
    main_file: Optional[str]


class InspectorAgent:
    """
    Analyzes GitHub repositories to extract code structure and entry points.
    
    This agent clones repositories, traverses file trees, and identifies
    main functions, classes, and data structures.
    """
    
    def __init__(self, llm_client: Optional[Any] = None, log_callback: Optional[Callable[[str], None]] = None):
        """
        Initialize the Inspector Agent.
        
        Args:
            llm_client: Optional LLM client for enhanced code understanding.
            log_callback: Optional callback function for logging messages.
        """
        self.llm_client = llm_client
        self.log = log_callback or (lambda msg: print(f"[Inspector] {msg}"))
        self.temp_base = tempfile.gettempdir()
        
    def clone_repository(self, github_url: str, session_id: str) -> str:
        """
        Clone a GitHub repository to a temporary directory.
        
        Args:
            github_url: The GitHub URL to clone.
            session_id: Unique session identifier.
            
        Returns:
            Path to the cloned repository.
        """
        repo_path = os.path.join(self.temp_base, f"repo_{session_id}")
        
        # Clean up existing directory if present
        if os.path.exists(repo_path):
            self.log(f"Cleaning up existing directory: {repo_path}")
            shutil.rmtree(repo_path)
            
        self.log(f"Cloning repository from {github_url}...")
        
        try:
            git.Repo.clone_from(github_url, repo_path, depth=1)
            self.log(f"✓ Repository cloned successfully to {repo_path}")
            return repo_path
        except Exception as e:
            self.log(f"✗ Clone failed: {e}")
            raise RuntimeError(f"Failed to clone repository: {e}")
    
    def traverse_file_tree(self, repo_path: str) -> List[str]:
        """
        Find all Python files in the repository.
        
        Args:
            repo_path: Path to the repository.
            
        Returns:
            List of Python file paths relative to repo root.
        """
        self.log("Traversing file tree...")
        python_files = []
        
        for root, dirs, files in os.walk(repo_path):
            # Skip hidden directories and common non-source dirs
            dirs[:] = [d for d in dirs if not d.startswith('.') 
                       and d not in ['__pycache__', 'node_modules', 'venv', '.venv', 'env']]
            
            for file in files:
                if file.endswith('.py'):
                    rel_path = os.path.relpath(os.path.join(root, file), repo_path)
                    python_files.append(rel_path)
                    
        self.log(f"✓ Found {len(python_files)} Python files")
        return python_files
    
    def _extract_function_info(self, node: ast.FunctionDef, file_path: str) -> FunctionInfo:
        """Extract information from a function AST node."""
        # Get parameters
        params = []
        for arg in node.args.args:
            param = {"name": arg.arg}
            if arg.annotation:
                param["type"] = ast.unparse(arg.annotation)
            params.append(param)
            
        # Get return type
        return_type = None
        if node.returns:
            return_type = ast.unparse(node.returns)
            
        # Build signature
        param_strs = []
        for p in params:
            if "type" in p:
                param_strs.append(f"{p['name']}: {p['type']}")
            else:
                param_strs.append(p['name'])
        signature = f"def {node.name}({', '.join(param_strs)})"
        if return_type:
            signature += f" -> {return_type}"
            
        return FunctionInfo(
            name=node.name,
            file_path=file_path,
            line_number=node.lineno,
            signature=signature,
            docstring=ast.get_docstring(node),
            return_type=return_type,
            parameters=params
        )
    
    def _extract_class_info(self, node: ast.ClassDef, file_path: str) -> ClassInfo:
        """Extract information from a class AST node."""
        methods = []
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                methods.append(item.name)
                
        bases = [ast.unparse(base) for base in node.bases]
        
        return ClassInfo(
            name=node.name,
            file_path=file_path,
            line_number=node.lineno,
            docstring=ast.get_docstring(node),
            methods=methods,
            bases=bases
        )
    
    def analyze_python_file(self, repo_path: str, file_path: str) -> Dict[str, Any]:
        """
        Analyze a single Python file for functions and classes.
        
        Args:
            repo_path: Path to the repository root.
            file_path: Relative path to the Python file.
            
        Returns:
            Dictionary with functions and classes found.
        """
        full_path = os.path.join(repo_path, file_path)
        functions = []
        classes = []
        has_main = False
        
        try:
            with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    if not node.name.startswith('_') or node.name == '__init__':
                        functions.append(self._extract_function_info(node, file_path))
                elif isinstance(node, ast.ClassDef):
                    if not node.name.startswith('_'):
                        classes.append(self._extract_class_info(node, file_path))
                        
            # Check for if __name__ == "__main__":
            for node in ast.walk(tree):
                if isinstance(node, ast.If):
                    try:
                        test = ast.unparse(node.test)
                        if "__name__" in test and "__main__" in test:
                            has_main = True
                            break
                    except:
                        pass
                        
        except Exception as e:
            self.log(f"  Warning: Could not parse {file_path}: {e}")
            
        return {
            "functions": functions,
            "classes": classes,
            "has_main": has_main
        }
    
    def find_requirements(self, repo_path: str) -> Optional[str]:
        """Find requirements.txt or pyproject.toml in the repository."""
        for req_file in ['requirements.txt', 'pyproject.toml', 'setup.py', 'environment.yml']:
            path = os.path.join(repo_path, req_file)
            if os.path.exists(path):
                self.log(f"✓ Found dependency file: {req_file}")
                return req_file
        self.log("  No dependency file found")
        return None
    
    def find_main_file(self, repo_path: str, python_files: List[str]) -> Optional[str]:
        """Identify the likely main/entry point file."""
        # Priority order for main files
        candidates = ['main.py', 'app.py', 'run.py', '__main__.py', 'cli.py']
        
        for candidate in candidates:
            for py_file in python_files:
                if py_file.endswith(candidate):
                    self.log(f"✓ Identified main file: {py_file}")
                    return py_file
                    
        # Check for files with if __name__ == "__main__"
        for py_file in python_files:
            result = self.analyze_python_file(repo_path, py_file)
            if result.get("has_main"):
                self.log(f"✓ Found entry point in: {py_file}")
                return py_file
                
        return None
    
    def generate_summary(self, repo_path: str) -> RepoSummary:
        """
        Generate a complete summary of the repository.
        
        Args:
            repo_path: Path to the cloned repository.
            
        Returns:
            RepoSummary with all extracted information.
        """
        self.log("Generating repository summary...")
        
        python_files = self.traverse_file_tree(repo_path)
        requirements = self.find_requirements(repo_path)
        main_file = self.find_main_file(repo_path, python_files)
        
        all_functions = []
        all_classes = []
        entry_points = []
        
        for py_file in python_files:
            self.log(f"  Analyzing: {py_file}")
            result = self.analyze_python_file(repo_path, py_file)
            all_functions.extend(result["functions"])
            all_classes.extend(result["classes"])
            
            if result["has_main"]:
                entry_points.append(py_file)
                
        self.log(f"✓ Found {len(all_functions)} functions, {len(all_classes)} classes")
        
        return RepoSummary(
            repo_path=repo_path,
            entry_points=entry_points,
            python_files=python_files,
            functions=all_functions,
            classes=all_classes,
            requirements_file=requirements,
            main_file=main_file
        )
    
    def to_json(self, summary: RepoSummary) -> str:
        """Convert summary to JSON for storage."""
        data = {
            "repo_path": summary.repo_path,
            "entry_points": summary.entry_points,
            "python_files": summary.python_files,
            "functions": [asdict(f) for f in summary.functions],
            "classes": [asdict(c) for c in summary.classes],
            "requirements_file": summary.requirements_file,
            "main_file": summary.main_file
        }
        return json.dumps(data, indent=2)
    
    def inspect(self, github_url: str, session_id: str) -> RepoSummary:
        """
        Main entry point: Clone and analyze a GitHub repository.
        
        Args:
            github_url: URL of the GitHub repository.
            session_id: Unique session identifier.
            
        Returns:
            Complete RepoSummary of the repository.
        """
        self.log("=" * 50)
        self.log("Starting repository inspection")
        self.log("=" * 50)
        
        repo_path = self.clone_repository(github_url, session_id)
        summary = self.generate_summary(repo_path)
        
        # Save summary to file
        summary_path = os.path.join(repo_path, "repo_summary.json")
        with open(summary_path, 'w') as f:
            f.write(self.to_json(summary))
        self.log(f"✓ Summary saved to {summary_path}")
        
        return summary


if __name__ == "__main__":
    # Test with a simple local inspection
    agent = InspectorAgent()
    
    # Example usage
    print("Inspector Agent initialized. Use inspect(github_url, session_id) to analyze a repo.")
