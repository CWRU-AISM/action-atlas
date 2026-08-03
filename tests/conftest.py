import sys
from pathlib import Path

# Make the repo root importable so tests can `from experiments... import ...`
# without an editable install.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
