"""
output.py
~~~~~~~~~
• ModelManifest – dict-subclass mirroring OutputDoc in analytic_schema.  
• save_model()  – one-shot convenience that pickles a model *and* writes
                   its accompanying manifest JSON.
Only stdlib + pandas.  Uses importlib.metadata for dependency capture.
"""

from __future__ import annotations
import datetime as _dt
import getpass, hashlib, json, os, pickle, platform, socket, sys
from pathlib import Path
from typing import Any, Dict, Tuple

import importlib.metadata as _im
import pandas as pd

from .loader import OUTPUT_SCHEMA
from .validator import validate_manifest, SchemaError

# ─────────────────────────────────────────────────────────────────────────────
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fd:
        for chunk in iter(lambda: fd.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def _library_versions() -> Dict[str, str]:
    wanted = {"scikit-learn", "pandas", "numpy", "tensorflow", "torch", "xgboost"}
    versions = {}
    for dist in _im.distributions():
        n = dist.metadata["Name"]
        if n and n.lower() in wanted:
            versions[n] = dist.version
    return versions

def _hardware_specs() -> Dict[str, str]:
    cpu = platform.processor() or platform.machine()
    ram = ""
    try:
        import psutil                      # noqa: S402  (optional)
        ram = f"{round(psutil.virtual_memory().total/2**30)} GB"
    except Exception:
        pass
    gpu = os.getenv("NVIDIA_VISIBLE_DEVICES", "")  # crude fallback
    return {"cpu": cpu, "gpu": gpu, "ram": ram}

def _now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")

# ─────────────────────────────────────────────────────────────────────────────
class ModelManifest(dict):
    """
    Manifest builder (similar to OutputDoc).
    Typical use:
        m = ModelManifest(model_type="RandomForest", ...)
        m.finalise(model_path)
        m.save("rf_manifest.json")
    """

    def __init__(self, *, model_type: str, **kwargs):
        super().__init__(**kwargs)
        self["model_type"]  = model_type
        self["export_dtg"]  = _now_iso()  # always set at creation
        self._finalised     = False

    # --------------------------------------------------------------------- #
    # Finalise & validate
    # --------------------------------------------------------------------- #
    def finalise(self, model_path: Path) -> None:
        if self._finalised:
            return
        self["model_file_hash"] = _sha256(model_path)

        # execution_environment auto-populate if not supplied
        self.setdefault("execution_environment", {
            "python_version":  platform.python_version(),
            "library_dependencies": _library_versions(),
            "operating_system": f"{platform.system()} {platform.release()}",
            "username": getpass.getuser(),
            "hardware_specs": _hardware_specs(),
        })

        validate_manifest(self)
        self._finalised = True

    # --------------------------------------------------------------------- #
    def save(self, path: Path | str, *, indent: int = 2) -> None:
        if not self._finalised:
            raise RuntimeError("Manifest must be finalised() before save()")
        Path(path).write_text(json.dumps(self, ensure_ascii=False, indent=indent),
                              encoding="utf-8")

# ─────────────────────────────────────────────────────────────────────────────
def save_model(
    model: Any,
    *,
    manifest: Dict[str, Any],
    directory: str | Path = ".",
    file_prefix: str | None = None
) -> Tuple[Path, Path]:
    """
    Convenience one-liner:
        model_path, manifest_path = save_model(clf, manifest=dict(...))
    • Pickles *model* using stdlib pickle (protocol-5 if available).
    • Populates/validates the manifest.
    • Returns paths to both artefacts.
    """
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)

    dtg   = _dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    mtype = manifest.get("model_type", getattr(model, "__class__", type(model)).__name__)
    stem  = file_prefix or f"{mtype}_{dtg}"

    model_path    = directory / f"{stem}.pkl"
    manifest_path = directory / f"{stem}_manifest.json"

    # 1) serialise model
    with model_path.open("wb") as fd:
        pickle.dump(model, fd, protocol=pickle.HIGHEST_PROTOCOL)

    # 2) build & validate manifest
    mani = ModelManifest(**manifest)     # inject user-supplied fields
    mani.finalise(model_path)
    mani.save(manifest_path)

    return model_path, manifest_path