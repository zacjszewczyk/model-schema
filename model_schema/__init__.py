"""Initializes the model_schema package and exposes its public API.

This file serves as the main entry point for the `model_schema` package.
It imports the core components from its submodules (`loader`, `validator`,
and `output`) and makes them available at the top-level of the package.
This creates a convenient and consistent public interface for users.

Example:
    >>> from model_schema import ModelManifest, validate_manifest
    >>> manifest = ModelManifest(model_type="RandomForest")
    >>> # ... build the manifest ...
    >>> validate_manifest(manifest)

"""
# Import key components from submodules to create the public package API.
from .loader    import OUTPUT_SCHEMA, SCHEMA_VERSION, SCHEMA_PATH
from .validator import validate_manifest, SchemaError
from .output    import save_model, ModelManifest

# The `__all__` variable explicitly declares the public names that should
# be imported when a client uses a wildcard import (e.g., `from
# model_schema import *`). This helps prevent namespace pollution and
# clearly defines the package's public contract.
__all__ = [
    "OUTPUT_SCHEMA", "SCHEMA_VERSION", "SCHEMA_PATH",
    "validate_manifest", "SchemaError",
    "save_model", "ModelManifest",
]