"""
model_schema package
~~~~~~~~~~~~~~~~~~~~
Public façade – mirrors analytic_schema’s API shape.

>>> from model_schema import save_model, ModelManifest, SchemaError
"""

from .loader    import OUTPUT_SCHEMA, SCHEMA_VERSION, SCHEMA_PATH
from .validator import validate_manifest, SchemaError
from .output    import save_model, ModelManifest

__all__ = [
    "OUTPUT_SCHEMA", "SCHEMA_VERSION", "SCHEMA_PATH",
    "validate_manifest", "SchemaError",
    "save_model", "ModelManifest",
]