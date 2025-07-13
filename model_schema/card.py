"""Markdown helpers for contracts & manifests.

This module converts:

1.  **The JSON contract** (schema) → a concise, human-readable *contract card*.
2.  **A single model manifest**   → a richly-formatted *model card*.

Both helpers return a **Markdown** string that can be written to disk, displayed
in notebooks, or pushed to documentation sites.

Typical use
-----------

import model_schema as msc
# Contract card
md = msc.contract_card()          # Uses the packaged schema by default
Path("contract.md").write_text(md, encoding="utf-8")

# Model card
mani = msc.ModelManifest(model_type="RandomForest", ...)
mani.finalise(Path("rf.pkl"))
card = msc.model_card(mani)
Path("rf_card.md").write_text(card, encoding="utf-8")
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

# Internal import – the packaged schema (used when no override supplied).
from .loader import OUTPUT_SCHEMA

# ─── Private helpers ────────────────────────────────────────────────────────

def _join_types(t: Sequence[str] | str | None) -> str:
    """Return a slash-separated, lowercase string of schema types."""
    if not t:
        return ""
    return "/".join(sorted(t if isinstance(t, (list, tuple)) else [t])).lower()


def _schema_fields_to_md(fields: Mapping[str, Any], level: int = 3) -> str:
    """Recursively render fields → Markdown."""
    md_parts: list[str] = []
    hdr = "#" * level
    for name in sorted(fields):  # deterministic
        meta      = fields[name]
        required  = meta.get("required", False)
        req_flag  = "(Required)" if required else "(Optional)"
        typestr   = _join_types(meta.get("type"))
        desc      = meta.get("description", "").strip()

        md_parts.append(f"{hdr} {name}. {req_flag}\n\n"
                        f"({typestr}). {desc}".rstrip())

        # Nested objects (recurse one level deeper)
        if "fields" in meta:
            md_parts.append(_schema_fields_to_md(meta["fields"], level + 1))
    return "\n\n".join(md_parts)


def _manifest_to_md(obj: Any, level: int = 3) -> str:
    """Recursively convert a manifest value → Markdown."""
    hdr  = "#" * level
    out: list[str] = []

    if isinstance(obj, Mapping):
        # Dict – iterate keys in sorted order for stability
        for key in sorted(obj):
            val = obj[key]
            out.append(f"{hdr} {key}\n")
            out.append(_manifest_to_md(val, level + 1))
    elif isinstance(obj, (list, tuple)):
        # Simple list → comma-separated string. Complex items handled individually.
        rendered = ", ".join(map(lambda x: json.dumps(x, ensure_ascii=False)
                                 if isinstance(x, (dict, list)) else str(x), obj))
        out.append(rendered or "--")
    else:
        # Scalar – just stringify
        out.append(str(obj) if obj is not None else "--")

    return "\n\n".join(out).rstrip()

# ─── Public API ─────────────────────────────────────────────────────────────

def contract_card(schema: Mapping[str, Any] | None = None) -> str:
    """Render a JSON schema contract → Markdown.

    Args:
        schema: The schema mapping.  If *None*, the packaged `OUTPUT_SCHEMA`
                is used.

    Returns:
        str: A Markdown string.
    """
    schema = schema or OUTPUT_SCHEMA
    title  = schema.get("title", "Model Schema")
    desc   = schema.get("description", "").strip()

    body   = _schema_fields_to_md(schema.get("fields", {}), level=3)

    return f"# {title}\n\n{desc}\n\n{body}\n"


def model_card(manifest: Mapping[str, Any]) -> str:
    """Render a model manifest → Markdown "model card".

    Args:
        manifest: A dictionary (or `ModelManifest`) containing model metadata.

    Returns:
        str: A Markdown string.
    """
    # Title & summary paragraph
    mtype  = manifest.get("model_type", "Model")
    title  = f"# {mtype}"
    summary_parts = [
        manifest.get("model_description", "").strip(),
        manifest.get("intended_use", "").strip(),
        manifest.get("limitations", "").strip(),
    ]
    summary = " ".join(p for p in summary_parts if p)

    # Key/Value section (everything – including the summary fields – for parity)
    body   = _manifest_to_md(manifest, level=3)

    return f"{title}\n\n{summary}\n\n{body}\n"