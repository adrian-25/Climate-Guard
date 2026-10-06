"""Root conftest.py — ensures the project root is on sys.path for all tests."""
import sys
from pathlib import Path

# Insert repo root so `import dl`, `import src`, etc. work without install
root = str(Path(__file__).resolve().parent)
if root not in sys.path:
    sys.path.insert(0, root)
