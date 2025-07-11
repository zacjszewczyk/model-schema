"""
validator.py
~~~~~~~~~~~~
Ultra-light JSON-Schema subset – identical philosophy to analytic_schema.
"""

from __future__ import annotations
import re
from typing import Any, Mapping, Tuple, Dict, Union, List

import pandas as pd

from .loader import OUTPUT_SCHEMA

class SchemaError(ValueError):
    ...

# ISO-8601 timestamp
_DT_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?(?:Z|[+\-]\d{2}:\d{2})$"
)
def _is_dtg(s: Any) -> bool:
    from datetime import datetime
    if not isinstance(s, str) or not _DT_RE.fullmatch(s):
        return False
    try:
        datetime.fromisoformat(s.replace("Z", "+00:00"))
        return True
    except ValueError:
        return False

# schema-type → Python class mapping
_TYPE: Dict[str, Union[type, Tuple[type, ...]]] = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "object": dict,
    "list": list,
    "dataframe": pd.DataFrame,
}

def _validate(val: Any, schema: Mapping[str, Any], *, path: str = "") -> None:
    """Recursive validator used by validate_manifest()"""
    stype = schema.get("type")
    if stype:
        allowed = stype if isinstance(stype, list) else [stype]
        py_ok = any(isinstance(val, _TYPE[t]) for t in allowed if t in _TYPE)
        if not py_ok:
            raise SchemaError(f"{path or 'value'}: expected {allowed}, got {type(val).__name__}")

    if "enum" in schema and val not in schema["enum"]:
        raise SchemaError(f"{path}: '{val}' not in {schema['enum']}")

    if schema.get("format") == "date-time" and not _is_dtg(val):
        raise SchemaError(f"{path}: '{val}' is not ISO-8601")

    # object recursion
    want_obj = ("object" in (stype or [])) or "fields" in schema
    if want_obj and isinstance(val, dict):
        fields = schema.get("fields", {})
        req    = {k for k,meta in fields.items() if meta.get("required")}
        extra  = set(val) - set(fields)
        if not schema.get("additionalProperties", True) and extra:
            raise SchemaError(f"{path}: unexpected {sorted(extra)}")
        miss = req - set(val)
        if miss:
            raise SchemaError(f"{path}: missing {sorted(miss)}")
        for k,v in val.items():
            if k in fields:
                _validate(v, fields[k], path=f"{path}.{k}" if path else k)

    # list recursion
    if stype == "list" and isinstance(val, list):
        if "items" in schema:
            for i,item in enumerate(val):
                _validate(item, schema["items"], path=f"{path}[{i}]")
        elif "subtype" in schema:
            sub = {"type": [schema["subtype"]]}
            for i,item in enumerate(val):
                _validate(item, sub, path=f"{path}[{i}]")

def validate_manifest(doc: Mapping[str, Any]) -> None:
    """Public entry: raise SchemaError on any violation."""
    _validate(doc, OUTPUT_SCHEMA, path="manifest")
 