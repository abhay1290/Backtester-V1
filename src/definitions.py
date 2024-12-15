import os
from pathlib import Path

SRC_DIR: Path = Path(os.path.dirname(os.path.abspath(__file__)))
ROOT_DIR: Path = SRC_DIR.parent
DATA_DIR: Path = ROOT_DIR / "data"

# Files
SPX_INDEX_DATA: Path = DATA_DIR / "spx_index.csv"
SPX_FUTURE_DATA: Path = DATA_DIR / "spx_future.csv"
