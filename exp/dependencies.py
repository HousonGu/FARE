"""Source closure used to bind resume to this checkout."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def dependency_files():
    return sorted(p for folder in ["src/fare","exp","data_provider","backbones"] for p in (ROOT/folder).rglob("*.py") if "__pycache__" not in p.parts and "tests" not in p.parts)
