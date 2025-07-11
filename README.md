# Model Schema

Model Schema is a lightweight Python package for **saving machine-learning models** and **building fully-featured model manifests** based on a single, versioned JSON contract.  
The contract captures every detail required to reproduce, audit, and deploy a model—library versions, hardware specs, dataset hashes, hyper-parameters, metrics, and more.

## Table of Contents

* [**Description**](#description)  
* [**Dependencies**](#dependencies)  
* [**Installation**](#installation)  
* [**Usage**](#usage)  
* [**Project structure**](#project-structure)  
* [**Background and Motivation**](#background-and-motivation)  
* [**Contributing**](#contributing)  
* [**Contributors**](#contributors)  
* [**License**](#license)  

## Description

Model Schema centralises your model artefact’s metadata in **one authoritative JSON file**. From that contract it automatically:

- Serialises any picklable model under a predictable filename (`<model_type>_<dtg>.pkl`).  
- Builds an **incremental** manifest so you can attach metrics, dataset hashes, and hyper-parameters as they become available.  
- Collects the complete **execution environment** (Python version, installed libraries, OS, hardware specs, user, host).  
- Computes SHA-256 hashes for the full dataset, each split, and the final model file for tamper-evidence and reproducibility.  
- Performs deep schema validation (type checks, enums, ISO-8601 date-times, oneOf branches, and no extra fields).  
- Serialises the finished manifest—or a collection of manifests from hyper-parameter sweeps—to pretty-printed JSON ready for CI pipelines, model registries, or downstream consumers.

With zero dependencies beyond the standard library, Model Schema is ideal for air-gapped training clusters, on-prem MLOps pipelines, or any environment that needs a robust, self-contained provenance layer for models.

## Dependencies

This project depends **only on the Python standard library (≥3.8)**.  
If available, `importlib-metadata` (bundled with Python ≥3.10) is used to auto-collect installed package versions; otherwise that feature degrades gracefully.

## Installation

```bash
pip install model-schema
```

In your code:

```
import model_schema as msc
```

## Usage

Check out example_incremental_export.py for a guided walkthrough. That script trains several models, serialises each one, and emits a single JSON file containing every manifest.

```
from model_schema import ModelManifest, validate_manifest

# 1. Build the manifest incrementally while you train
mani = ModelManifest(
    model_type=“RandomForest”,
    model_version=“1.0.0”,
    model_description=“Random-Forest classifier for churn prediction”,
    intended_use=“Marketing churn prevention”,
    limitations=“May underperform with severe concept drift”,
    feature_names=[“age”, “income”, “tenure”, “plan”],
    target_variable=“churned”,
    random_seed=42,
    # …plus any other required contract fields…
)

# Add metrics as they become available
mani.add_metrics(split=“training”, accuracy=0.95, f1=0.94)
mani.add_metrics(split=“validation”, accuracy=0.91, f1=0.88)

# 2. Save your trained model (pickle or joblib—your call)
model_path = “RandomForest_2025-07-11T12-00-00Z.pkl”
with open(model_path, “wb”) as fd:
    pickle.dump(rf_model, fd)

# 3. Finalise → inject environment + model hash → validate
mani.finalise(model_path)

# 4. Persist the manifest
mani.save_manifest(“rf_manifest.json”)
```

## Project structure

```
model-schema/            # Project repository
├── model_schema/        # Package
│   ├── __init__.py
│   ├── model_schema.py  # Loader, validator, ModelManifest helper
│   ├── output.py        # Incremental save helpers (optional extras)
│   └── model_schema.json# Single source-of-truth contract
│
├── tests/
│   └── test_model_schema.py
│
├── example_incremental_export.py
│
├── README.md            # This file
├── LICENSE.md           # Project license
├── Makefile             # Build & packaging helpers
└── pyproject.toml
```

## Background and Motivation

In ML ops, reproducibility and provenance are paramount. Months after a model is deployed you may need to:
	•	Re-train on new data with identical hyper-parameters.
	•	Verify which CUDA driver or library version was used.
	•	Compare current performance with historic baselines.

Without a rigorous manifest these tasks become guess-work.
Model Schema solves that by elevating model metadata to a first-class, contract-driven artefact:
	•	Uniformity – Every model export shares the same field names, types, and defaults.
	•	Reliability – Fail-fast validation prevents missing or malformed metadata.
	•	Traceability – Hashes across inputs, splits, and the model file itself enable full audit trails.
	•	Simplicity – Pure stdlib design keeps dependencies minimal and works in restricted environments.

By abstracting away boilerplate, you can focus on building better models while ensuring your pipeline remains robust, maintainable, and easy to integrate.

## Contributing

Contributions are welcome from everyone, regardless of role or experience.

There are no special system requirements for contributing. To edit via the web:
	1.	Click the repository’s “Web IDE” button.
	2.	Make your change (one logical change per commit).
	3.	Click “Commit…” → provide a descriptive message.
	4.	Choose “Create a new branch” (recommended name: first.last).
	5.	Click “Commit”, then open a merge request.

You can also contribute locally by forking/cloning the repo, creating a branch,
pushing your changes, and opening a PR.

## Contributors

This section lists project contributors.
When you submit a merge request, append your name to the list below and (optionally) the sections you contributed.

* Creator: Zachary Szewczyk

## License

This project is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.
See the full text in LICENSE.md. In short, you may remix and share non-commercial derivatives of this work, provided you attribute the original author and license your contributions under the same terms.