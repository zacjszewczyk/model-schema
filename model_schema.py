#!/usr/bin/env python3
"""A self-contained module for creating and validating model manifests.

This module provides a robust, dependency-free toolkit for creating structured,
version-controlled metadata for machine learning models. It is designed to be
dropped into any Python environment (version 3.8 or higher) without requiring
external packages.

The core functionality includes:
- Loading a `model_schema.json` contract which acts as the single
  source of truth for the manifest structure.
- A minimal, yet powerful, JSON Schema validator that checks for type
  correctness, required fields, and other structural constraints.
- A convenience class, `ModelManifest`, that simplifies the process of
  building a valid manifest by incrementally gathering metadata and
- automatically injecting execution environment details and file hashes.

Public-facing components:
- `validate_manifest(doc)`: Validates a dictionary against the schema.
- `ModelManifest(**seed)`: A dict subclass for building a manifest.
- `SchemaError`: Custom exception raised on validation failure.
"""

from __future__ import annotations

# Standard library imports for core functionalities like file handling,
# data serialization, hashing, and system introspection.
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

# The `importlib.metadata` module, available in Python 3.8+, is used to
# inspect the installed packages in the environment. A fallback to `None`
# is provided for older versions, although the project requires >=3.8.
try:
    # Python >=3.8 is the primary target.
    import importlib.metadata as _importlib_metadata
except ModuleNotFoundError:  # pragma: no cover
    # This fallback ensures the code does not crash on older interpreters,
    # though environment capture will be disabled.
    _importlib_metadata = None  # type: ignore

# Resolve the path to the `model_schema.json` contract file.
# This approach ensures that the schema can be located reliably, regardless
# of the script's current working directory.
_PKG_DIR      = pathlib.Path(__file__).parent.resolve()
SCHEMA_PATH   = _PKG_DIR / "model_schema.json"

# Load the raw schema from the JSON file at module import time.
# This makes the schema immediately available for validation and raises
# an error early if the contract file is missing or malformed.
try:
    with SCHEMA_PATH.open(encoding="utf-8") as fd:
        _SCHEMA_RAW = json.load(fd)
except FileNotFoundError as exc:
    raise FileNotFoundError(
        f"Contract file not found: {SCHEMA_PATH}"
    ) from exc
except json.JSONDecodeError as exc:
    raise ValueError(f"Contract file is not valid JSON: {exc}") from exc

# Extract the schema version and the 'output' schema definition.
# A deep copy is used for `OUTPUT_SCHEMA` to prevent any modifications
# to it from accidentally altering the original raw schema object.
SCHEMA_VERSION = _SCHEMA_RAW.get("version", "UNKNOWN")
OUTPUT_SCHEMA: Dict[str, Any] = copy.deepcopy(_SCHEMA_RAW["output"])


class SchemaError(ValueError):
    """Raised on any manifest-validation failure."""


# A compiled regular expression for validating ISO 8601 datetime strings.
# Compiling the regex improves performance for repeated validation checks.
_DT_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}"    # YYYY-MM-DDThh:mm:ss
    r"(?:\.\d{1,6})?"                         # optional .microseconds
    r"(?:Z|[+-]\d{2}:\d{2})$"                 # Z or timezone offset +/-HH:MM
)


def _is_dt(s: Any) -> bool:
    """Check if a string is a valid ISO 8601 datetime.

    This function first uses a regular expression for a fast check of the
    string format and then attempts to parse it using `datetime.fromisoformat`
    for a definitive validation.

    Args:
        s (Any): The value to check.

    Returns:
        bool: True if `s` is a valid ISO 8601 datetime string, False otherwise.
    """
    if not isinstance(s, str) or not _DT_RE.match(s):
        return False
    try:
        # The 'Z' suffix for UTC is not directly supported by `fromisoformat`
        # until Python 3.11, so it's replaced with a compatible offset.
        _dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
        return True
    except ValueError:
        return False


# A dispatch dictionary mapping JSON schema type names to their corresponding
# Python types. This allows the validator to perform `isinstance` checks.
_TYPE_DISPATCH: Dict[str, Union[type, Tuple[type, ...]]] = {
    "string":  str,
    "integer": int,
    "number":  (int, float),  # 'number' can be an integer or a float.
    "object":  dict,
    "list":    list,
    "boolean": bool,
}


def _validate(val: Any, schema: Mapping[str, Any], *, path: str = "") -> None:
    """Recursively validate a value against a schema definition.

    This function implements a minimal subset of the JSON Schema specification,
    providing core validation capabilities for types, enums, formats, and
    nested structures like objects and lists.

    Args:
        val (Any): The data value to be validated.
        schema (Mapping[str, Any]): The schema rules to apply to `val`.
        path (str): The dot-notation path to the current value, used for
                    generating clear and specific error messages.

    Raises:
        SchemaError: If `val` fails any validation rule defined in `schema`.
    """
    # 1. Validate the value's type against the schema's 'type' specification.
    # It supports both a single type string and a list of allowed types.
    typespec = schema.get("type")
    if typespec:
        allowed = typespec if isinstance(typespec, list) else [typespec]
        if not any(
            isinstance(val, _TYPE_DISPATCH.get(t, ())) for t in allowed
        ):
            raise SchemaError(
                f"{path or 'value'}: expected {allowed}, got {type(val).__name__}"
            )

    # 2. Validate that the value is one of the allowed values in the 'enum'.
    if "enum" in schema and val not in schema["enum"]:
        raise SchemaError(f"{path}: '{val}' not in {schema['enum']}")

    # 3. Validate special string formats, such as 'date-time'.
    if schema.get("format") == "date-time" and not _is_dt(val):
        raise SchemaError(f"{path}: '{val}' is not ISO-8601 date-time")

    # 4. Handle 'oneOf' schema alternatives. The value must validate
    # successfully against at least one of the schemas in the 'oneOf' list.
    if "oneOf" in schema:
        errs = []
        for alt in schema["oneOf"]:
            try:
                _validate(val, alt, path=path)
                break  # The first schema that validates successfully is sufficient.
            except SchemaError as exc:
                errs.append(str(exc))
        else:
            # This block executes if the loop completes without a `break`,
            # meaning no alternative schemas were satisfied.
            raise SchemaError(
                f"{path}: does not satisfy any allowed schema in oneOf\n" +
                "\n".join(f"  {e}" for e in errs)
            )

    # 5. Recursively validate dictionary objects.
    if isinstance(val, dict):
        fields = schema.get("fields", {})

        # Check for missing required fields.
        required = {k for k, meta in fields.items() if meta.get("required")}
        missing  = required - val.keys()
        if missing:
            raise SchemaError(f"{path}: missing required {sorted(missing)}")

        # Check for unexpected fields if `additionalProperties` is false.
        if not schema.get("additionalProperties", True):
            extra = set(val) - set(fields)
            if extra:
                raise SchemaError(f"{path}: unexpected fields {sorted(extra)}")

        # Recursively validate each field in the dictionary.
        for k, v in val.items():
            if k in fields:
                _validate(v, fields[k], path=f"{path}.{k}" if path else k)

    # 6. Recursively validate list items.
    if isinstance(val, list):
        if "items" in schema:
            item_schema = schema["items"]
        elif "subtype" in schema:
            # 'subtype' is a non-standard but convenient shorthand for a simple
            # list where all items must be of the same basic type.
            item_schema = {"type": [schema["subtype"]]}
        else:
            item_schema = None

        # If a schema for items is defined, recursively validate each item.
        if item_schema:
            for i, itm in enumerate(val):
                _validate(itm, item_schema, path=f"{path}[{i}]")


def validate_manifest(doc: Mapping[str, Any]) -> None:
    """Validate a document against the global `OUTPUT_SCHEMA`.

    This function serves as the main public entry point for validation. It
    initiates the recursive validation process for a given manifest document.

    Args:
        doc (Mapping[str, Any]): The model manifest dictionary to validate.

    Raises:
        SchemaError: If the `doc` violates any rule in the loaded schema.
    """
    _validate(doc, OUTPUT_SCHEMA, path="manifest")


def _gather_environment() -> Dict[str, Any]:
    """Collect comprehensive details about the execution environment.

    This function captures software and hardware context to ensure that the
    conditions under which a model was trained are fully documented. This is
    critical for reproducibility.

    Returns:
        Dict[str, Any]: A dictionary containing environment details such as
                        Python version, installed libraries, OS, user, and
                        hardware specifications.
    """
    libs: Dict[str, str] = {}
    if _importlib_metadata:
        # Iterate through all installed distributions and record their names
        # and versions. This is crucial for recreating the environment.
        for dist in _importlib_metadata.distributions():  # pragma: no cover
            libs[dist.metadata["Name"]] = dist.version
    return {
        "python_version":     platform.python_version(),
        "library_dependencies": libs,
        "operating_system":   f"{platform.system()} {platform.release()}",
        "username":           (getpass.getuser() if hasattr(getpass, "getuser") else "unknown_user"),
        "hardware_specs": {
            "cpu": platform.processor() or "unknown_cpu",
            "gpu": os.getenv("CUDA_VISIBLE_DEVICES", "none"),
            # System configuration values for memory are not available on all
            # platforms (e.g., Windows), so a check is necessary.
            "ram": f"{round(os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / 1e9, 1)} GB"  # type: ignore
                   if hasattr(os, "sysconf") else "unknown",
        },
    }


def _sha256_file(path: pathlib.Path, chunk: int = 1 << 20) -> str:
    """Compute the SHA-256 hash of a file efficiently.

    It reads the file in chunks to avoid loading the entire file into memory,
    making it suitable for hashing very large model files.

    Args:
        path (pathlib.Path): The path to the file to be hashed.
        chunk (int): The size of each chunk to read from the file, in bytes.
                     Defaults to 1MB.

    Returns:
        str: The hexadecimal representation of the file's SHA-256 hash.
    """
    h = hashlib.sha256()
    with path.open("rb") as fd:
        while True:
            blk = fd.read(chunk)
            if not blk:
                break
            h.update(blk)
    return h.hexdigest()


class ModelManifest(dict):
    """A dictionary subclass that helps build a valid model manifest.

    This class acts as a convenience wrapper around a standard dictionary. It
    provides methods to incrementally add data and streamlines the finalization
    process by automatically injecting computed values like the model file
    hash and execution environment details before running validation.
    """

    def __init__(self, **seed: Any) -> None:
        """Initialize the manifest with a set of seed values.

        Args:
            **seed: Any key-value pairs that are already known at the time of
                    instantiation. These typically include user-defined
                    metadata like model description, hyperparameters, or
                    dataset details.
        """
        super().__init__(**seed)
        # An export timestamp is set by default upon creation. This ensures
        # that the manifest and any associated model files share the same
        # consistent timestamp for easier grouping and identification.
        self.setdefault(
            "export_dtg",
            _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
        )

    def add_metrics(self, *, split: str, **metrics: float) -> None:
        """Append or overwrite a set of metrics for a specific data split.

        This helper method provides a structured way to add performance metrics
        for different dataset splits like 'training', 'validation', or 'test'.

        Example:
            >>> mani = ModelManifest(model_type="RF")
            >>> mani.add_metrics(split="validation", accuracy=0.91, f1=0.88)

        Args:
            split (str): The name of the data split. Must be one of
                         'training', 'validation', or 'test'.
            **metrics (float): Arbitrary keyword arguments representing the
                               metric names and their float values.

        Raises:
            ValueError: If the `split` name is not one of the allowed values.
        """
        if split not in ("training", "validation", "test"):
            raise ValueError("split must be one of {'training','validation','test'}")
        self.setdefault("metrics", {})
        self["metrics"].setdefault(split, {}).update(metrics)

    def finalise(self, model_path: Union[str, pathlib.Path]) -> None:
        """Complete the manifest and run validation.

        This is the final step in the manifest creation process. It populates
        auto-generated fields, such as the model file's hash and the full
        execution environment, and then triggers a full validation against
        the schema to ensure the manifest is complete and correct.

        Args:
            model_path (Union[str, pathlib.Path]): The path to the saved,
                serialized model file.

        Raises:
            FileNotFoundError: If the `model_path` does not exist or is not
                               a file.
            SchemaError: If the completed manifest fails validation.
        """
        path_obj = pathlib.Path(model_path)
        if not path_obj.is_file():
            raise FileNotFoundError(f"Model file does not exist: {model_path}")

        # The execution environment is gathered and attached only if it hasn't
        # already been provided by the user.
        self.setdefault("execution_environment", _gather_environment())

        # The SHA-256 hash of the model file is computed and added.
        self["model_file_hash"] = _sha256_file(path_obj)

        # A full validation is run against the schema, which will raise a
        # `SchemaError` on any failure.
        validate_manifest(self)

    def save_manifest(self, path: Union[str, pathlib.Path], *, indent: int = 2) -> None:
        """Write the finalized manifest to a JSON file.

        Args:
            path (Union[str, pathlib.Path]): The destination file path.
            indent (int): The number of spaces to use for indentation to
                          pretty-print the JSON output. Defaults to 2.
        """
        pathlib.Path(path).write_text(
            json.dumps(self, ensure_ascii=False, indent=indent),
            encoding="utf-8"
        )


def _main(argv: list[str] | None = None) -> None:  # pragma: no cover
    """Provide a command-line interface for validating manifest files.

    This function allows the module to be run as a script from the terminal,
    offering a quick way to check if a given JSON manifest file conforms to
    the schema.

    Example:
        $ python -m model_schema --file my_model_manifest.json

    Args:
        argv (list[str] | None): A list of command-line arguments, primarily
                                 for testing purposes. If None, `sys.argv`
                                 is used.
    """
    import argparse
    import textwrap
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
        print("Manifest valid.")
    except SchemaError as exc:
        print("Manifest INVALID:\n", exc)
        sys.exit(2)


# This block allows the script to be executed directly from the command line.
if __name__ == "__main__":  # pragma: no cover
    _main()