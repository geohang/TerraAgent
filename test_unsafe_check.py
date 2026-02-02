"""Test UNSAFE installation check"""
import sys
import os

# Change to project directory
os.chdir(r"c:\Users\HChen8\OneDrive - University of Iowa\Documents\GitHub\TerraAgent")
sys.path.insert(0, os.getcwd())

# Test from source module
from src.science_flood import check_unsafe_installation

status = check_unsafe_installation()
print("=== Testing from src/science_flood.py ===")
print(f"UNSAFE installed: {status['installed']}")
print(f"Data ready: {status['data_ready']}")
print(f"Mode: {status['mode']}")
print(f"Data dir: {status['data_dir']}")

# Test the data directory detection
from pathlib import Path
print("\n=== Testing _find_data_dir logic ===")
candidates = [
    Path("src/science_flood.py").resolve().parent.parent / "data" / "unsafe",
    Path("generated_flood_app.py").resolve().parent / "data" / "unsafe",
    Path.cwd() / "data" / "unsafe",
]
for c in candidates:
    print(f"  {c} -> exists: {c.exists()}")
