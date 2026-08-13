# =====================================================================
# FILE: conftest.py
# =====================================================================
"""
pytest configuration file.
Adds the project root to Python path so tests can import src modules.
"""

import os
import sys

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Also add src to path
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Print path for debugging (optional)
# print(f"Python path includes: {project_root}")
