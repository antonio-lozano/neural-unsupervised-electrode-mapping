"""Pytest configuration file."""
import sys
from pathlib import Path

# Add the repository root to Python path so tests can import modules
repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
