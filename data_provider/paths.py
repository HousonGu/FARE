"""Paths for the standalone FARE reproduction checkout."""
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = Path(os.environ.get("FARE_RAW_ROOT", str(ROOT / "data" / "raw"))).expanduser().resolve()
RESULT_ROOT = Path(os.environ.get("FARE_RESULTS_ROOT", str(ROOT / "results"))).expanduser().resolve()
