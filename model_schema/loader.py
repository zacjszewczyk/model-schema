"""
loader.py
~~~~~~~~~
Loads *model_schema.json* (single source of truth).
"""

import json
from pathlib import Path
import copy

_SCRIPT_DIR   = Path(__file__).parent.resolve()
SCHEMA_PATH   = _SCRIPT_DIR / "model_schema.json"

try:
    with SCHEMA_PATH.open(encoding="utf-8") as fd:
        _SCHEMA_RAW = json.load(fd)
except FileNotFoundError as exc:
    raise FileNotFoundError(f"Contract not found: {SCHEMA_PATH}") from exc
except json.JSONDecodeError as exc:
    raise ValueError(f"Invalid JSON in {SCHEMA_PATH}: {exc}") from exc

SCHEMA_VERSION = _SCHEMA_RAW.get("version", "UNKNOWN")
OUTPUT_SCHEMA  = copy.deepcopy(_SCHEMA_RAW["output"])          # deep copy