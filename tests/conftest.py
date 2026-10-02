import sys
from pathlib import Path

# backtest.py and src/ live at the project root, one level up from tests/.
# Insert it into sys.path so `import backtest` / `from src...` resolve
# regardless of how pytest was invoked (bare `pytest`, `python -m pytest`,
# from a different cwd, etc).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
