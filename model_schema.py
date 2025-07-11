#!/usr/bin/env python3
"""
model_schema.py
~~~~~~~~~~~~~~~

Single-file helper that

1. **Loads** the `model_schema.json` contract (single source-of-truth).
2. Implements a **minimal JSON-Schema validator** (types, enums, date-times,
   required keys, additionalProperties, homogeneous lists, oneOf).
3. Exposes public helpers:
   • :func:`validate_manifest`  – deep-validate any dict against OUTPUT_SCHEMA.  
   • :class:`ModelManifest`     – convenience builder that auto-populates
     execution-environment, hashes the model file, and calls validation.

The file is entirely **self-contained** (standard-library only) so it can be
vendor-dropped into an air-gapped training environment.
"""

from __future__ import annotations

# ─── stdlib ────────────────────────────────────────────────────────────────
import copy
import datetime as _dt
import getpass
import hashlib
import json
import os
import pathlib
import platform
import re
import socket
import sys
import types
import uuid
from typing import Any, Dict, Mapping, Tuple, Union

try:
    # Python ≥3.10
    import importlib.metadata as _importlib_metadata
except ModuleNotFoundError:   # pragma: no cover
    _importlib_metadata = None     # type: ignore

# ─── Locate & load the contract ────────────────────────────────────────────
_PKG_DIR     = pathlib.Path(__file__).parent.resolve()
SCHEMA_PATH  = _PKG_DIR / "model_schema.json"

try:
    with SCHEMA_PATH.open(encoding="utf-8") as fd:
        _SCHEMA_RAW = json.load(fd)
except FileNotFoundError as exc:
    raise FileNotFoundError(
        f"Contract file not found: {SCHEMA_PATH}"
    ) from exc
except json.JSONDecodeError as exc:
    raise ValueError(f"Contract file is not valid JSON: {exc}") from exc

SCHEMA_VERSION = _SCHEMA_RAW.get("version", "UNKNOWN")
OUTPUT_SCHEMA: Dict[str, Any] = copy.deepcopy(_SCHEMA_RAW["output"])

# ─── Validation engine (subset of JSON-Schema) ─────────────────────────────
class SchemaError(ValueError):
    """Raised on any manifest-validation failure."""

_DT_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}"     # YYYY-MM-DDThh:mm:ss
    r"(?:\.\d{1,6})?"                            # optional .microseconds
    r"(?:Z|[+-]\d{2}:\d{2})$"                    # Z or ±HH:MM
)

def _is_dt(s: Any) -> bool:
    if not isinstance(s, str) or not _DT_RE.match(s):
        return False
    try:
        _dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
        return True
    except ValueError:
        return False

_TYPE_DISPATCH: Dict[str, Union[type, Tuple[type, ...]]] = {
    "string":  str,
    "integer": int,
    "number":  (int, float),
    "object":  dict,
    "list":    list,
    "boolean": bool,
}

def _validate(val: Any, schema: Mapping[str, Any], *, path: str = "") -> None:
    """
    Recursive validator that understands:
    * type (incl. unions)
    * enum
    * format: date-time
    * object   → fields / required / additionalProperties
    * list     → subtype or items
    * oneOf
    """
    # 1) Type ---------------------------------------------------------------
    typespec = schema.get("type")
    if typespec:
        allowed = typespec if isinstance(typespec, list) else [typespec]
        if not any(
            isinstance(val, _TYPE_DISPATCH.get(t, ())) for t in allowed
        ):
            raise SchemaError(
                f"{path or 'value'}: expected {allowed}, got {type(val).__name__}"
            )

    # 2) Enum ---------------------------------------------------------------
    if "enum" in schema and val not in schema["enum"]:
        raise SchemaError(f"{path}: '{val}' not in {schema['enum']}")

    # 3) format: date-time --------------------------------------------------
    if schema.get("format") == "date-time" and not _is_dt(val):
        raise SchemaError(f"{path}: '{val}' is not ISO-8601 date-time")

    # 4) oneOf --------------------------------------------------------------
    if "oneOf" in schema:
        errs = []
        for alt in schema["oneOf"]:
            try:
                _validate(val, alt, path=path)
                break         # first success wins
            except SchemaError as exc:
                errs.append(str(exc))
        else:
            raise SchemaError(
                f"{path}: does not satisfy any allowed schema in oneOf\n" +
                "\n".join(f"  {e}" for e in errs)
            )

    # 5) object recursion ---------------------------------------------------
    if isinstance(val, dict):
        fields = schema.get("fields", {})
        required = {k for k,meta in fields.items() if meta.get("required")}
        missing  = required - val.keys()
        if missing:
            raise SchemaError(f"{path}: missing required {sorted(missing)}")
        if not schema.get("additionalProperties", True):
            extra = set(val) - set(fields)
            if extra:
                raise SchemaError(f"{path}: unexpected fields {sorted(extra)}")
        for k,v in val.items():
            if k in fields:
                _validate(v, fields[k], path=f"{path}.{k}" if path else k)

    # 6) list recursion -----------------------------------------------------
    if isinstance(val, list):
        if "items" in schema:
            item_schema = schema["items"]
        elif "subtype" in schema:
            item_schema = {"type": [schema["subtype"]]}
        else:
            item_schema = None
        if item_schema:
            for i, itm in enumerate(val):
                _validate(itm, item_schema, path=f"{path}[{i}]")

# ─── Public helpers ────────────────────────────────────────────────────────
def validate_manifest(doc: Mapping[str, Any]) -> None:
    """Raise :class:`SchemaError` if *doc* violates the contract."""
    _validate(doc, OUTPUT_SCHEMA, path="manifest")

# ─── Execution-environment capture ─────────────────────────────────────────
def _gather_environment() -> Dict[str, Any]:
    libs: Dict[str, str] = {}
    if _importlib_metadata:
        for dist in _importlib_metadata.distributions():          # pragma: no cover
            libs[dist.metadata["Name"]] = dist.version
    return {
        "python_version":    platform.python_version(),
        "library_dependencies": libs,
        "operating_system":  f"{platform.system()} {platform.release()}",
        "username":          (getpass.getuser() if hasattr(getpass, "getuser") else "unknown_user"),
        "hardware_specs": {
            "cpu": platform.processor() or "unknown_cpu",
            "gpu": os.getenv("CUDA_VISIBLE_DEVICES", "none"),
            "ram": f"{round(os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / 1e9,1)} GB"   # type: ignore
                    if hasattr(os, "sysconf") else "unknown",
        },
    }

def _sha256_file(path: pathlib.Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fd:
        while True:
            blk = fd.read(chunk)
            if not blk:
                break
            h.update(blk)
    return h.hexdigest()

# ─── ModelManifest convenience class ───────────────────────────────────────
class ModelManifest(dict):
    """
    Convenience wrapper that

    * Accumulates metadata while you train/evaluate.
    * Computes the **model file hash** and **execution environment**.
    * Validates itself against the JSON contract in :meth:`finalise`.
    """

    def __init__(self, **seed: Any) -> None:
        """
        Parameters
        ----------
        **seed
            Any contract-required or optional fields you already know at
            construction time (hyper-parameters, dataset hashes, etc.).
        """
        super().__init__(**seed)
        # record early DTG so filenames & manifest share the same timestamp
        self.setdefault(
            "export_dtg",
            _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
        )

    # ------------------------------------------------------------------ #
    # Incremental helpers so users can attach new fields easily
    # ------------------------------------------------------------------ #
    def add_metrics(self, *, split: str, **metrics: float) -> None:
        """
        Append / overwrite metrics for `training`, `validation`, or `test`.

        Example
        -------
        >>> mani.add_metrics(split="validation", accuracy=0.91, f1=0.88, ...)
        """
        if split not in ("training", "validation", "test"):
            raise ValueError("split must be one of {'training','validation','test'}")
        self.setdefault("metrics", {})
        self["metrics"].setdefault(split, {}).update(metrics)

    # ------------------------------------------------------------------ #
    def finalise(self, model_path: Union[str, pathlib.Path]) -> None:
        """
        Complete the manifest by

        * computing SHA-256 over the saved *model_path*,
        * injecting the execution environment, and
        * running full schema validation.
        """
        path_obj = pathlib.Path(model_path)
        if not path_obj.is_file():
            raise FileNotFoundError(f"Model file does not exist: {model_path}")

        # attach environment only once
        self.setdefault("execution_environment", _gather_environment())

        # attach file hash
        self["model_file_hash"] = _sha256_file(path_obj)

        # **Validate** (raises on error)
        validate_manifest(self)

    # ------------------------------------------------------------------ #
    def save_manifest(self, path: Union[str, pathlib.Path], *, indent: int = 2) -> None:
        """
        Write the manifest to *path* as pretty-printed JSON (UTF-8).
        """
        pathlib.Path(path).write_text(
            json.dumps(self, ensure_ascii=False, indent=indent),
            encoding="utf-8"
        )

# ─── CLI helper (optional) ────────────────────────────────────────────────
def _main(argv: list[str] | None = None) -> None:        # pragma: no cover
    import argparse, textwrap
    parser = argparse.ArgumentParser(
        prog="model_schema",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""
        Quick command-line validator for model manifests.

          $ python -m model_schema --file my_model_manifest.json
        """),
    )
    parser.add_argument("--file", required=True, help="Path to manifest JSON")
    args = parser.parse_args(argv)

    obj = json.loads(pathlib.Path(args.file).read_text(encoding="utf-8"))
    try:
        validate_manifest(obj)
        print("✔ Manifest valid.")
    except SchemaError as exc:
        print("✘ Manifest INVALID:\n", exc)
        sys.exit(2)

if __name__ == "__main__":        # pragma: no cover
    _main()