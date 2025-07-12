from pathlib import Path
from time import perf_counter
import hashlib, json, pickle, random, datetime as dt, math

import numpy as np
import pandas as pd
from sklearn.datasets   import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics    import precision_recall_fscore_support, accuracy_score
from sklearn.ensemble   import RandomForestClassifier, GradientBoostingClassifier
from xgboost            import XGBClassifier                     # optional extra lib

import model_schema as msc

# ---------------------------------------------------------------------------
# 1)  Common prep – dataset, splits, helper for metrics → dict
# ---------------------------------------------------------------------------
iris = load_iris(as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.25, random_state=42, stratify=iris.target
)

def _metric_dict(y_true, y_pred):
    p,r,f,_ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    return {
        "precision": round(p,4),
        "recall":    round(r,4),
        "f1":        round(f,4),
        "accuracy":  round(accuracy_score(y_true, y_pred),4),
    }

# Run-level bookkeeping
run_dtg   = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
run_seed  = 42
random.seed(run_seed); np.random.seed(run_seed)

run_manifest = {
    "run_dtg": run_dtg,
    "random_seed": run_seed,
    "models": []            # ← we’ll append per-model manifests here
}

export_dir = Path("model_exports"); export_dir.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# 2)  Iterate over N models you create during the session
# ---------------------------------------------------------------------------
MODELS = [
    ("RandomForest",       RandomForestClassifier(n_estimators=200, random_state=run_seed)),
    ("GradientBoosting",   GradientBoostingClassifier(random_state=run_seed)),
    ("XGBoost",            XGBClassifier(n_estimators=150, eval_metric="logloss", random_state=run_seed)),
]

for model_type, model in MODELS:
    # ── Train + time it ────────────────────────────────────────────────────
    tic = perf_counter()
    model.fit(X_train, y_train)
    train_dur = perf_counter() - tic

    # ── Evaluate ───────────────────────────────────────────────────────────
    y_hat_tr = model.predict(X_train)
    y_hat_te = model.predict(X_test)

    metrics = {
        "training":  _metric_dict(y_train, y_hat_tr),
        "test":      _metric_dict(y_test,  y_hat_te),
    }

    # ── Serialise model file (filename uses model_type + dtg) ──────────────
    fn_stem = f"{model_type}_{run_dtg}"
    model_path = export_dir / f"{fn_stem}.pkl"
    with model_path.open("wb") as fd:
        pickle.dump(model, fd, pickle.HIGHEST_PROTOCOL)

    # ── Build manifest incrementally ───────────────────────────────────────
    mani = msc.ModelManifest(                     # minimal required seed data
        model_type       = model_type,
        model_architecture = model_type,
        model_version    = "1.0.0",
        model_description= f"{model_type} for iris classification",
        intended_use     = "Demo – flower species prediction",
        limitations      = "Toy dataset; not for production",
        data_description = "UCI iris dataset",
        data_schema      = {c: str(t) for c,t in zip(iris.data.columns, ["float"]*4)},
        feature_names    = iris.data.columns.tolist(),
        target_variable  = "species",
        feature_engineering_pipeline = ["None – raw numeric features"],
        hyperparameters  = {
                                k: (None if isinstance(v, float) and math.isnan(v) else v)
                                for k, v in model.get_params().items()
                            },
        number_trials    = 1,
        dataset_size     = len(iris.data),
        dataset_hash     = hashlib.sha256(iris.data.to_json().encode()).hexdigest(),
        train_size       = len(X_train),
        test_size        = len(X_test),
        train_hash       = hashlib.sha256(X_train.to_json().encode()).hexdigest(),
        test_hash        = hashlib.sha256(X_test.to_json().encode()).hexdigest(),
        random_seed      = run_seed,
        metrics          = metrics,
        training_duration_seconds = round(train_dur, 3)
    )

    # Add any late fields (e.g., test duration if you time it separately)
    # mani["test_duration_seconds"] = 0.123

    # ── Finalise & validate now that model_path exists ─────────────────────
    mani.finalise(model_path)        # adds model_file_hash, fills env block, validates

    # Buffer manifest in the run-level container
    run_manifest["models"].append(mani)

# ---------------------------------------------------------------------------
# 3)  Optional: validate every manifest inside the joined doc for sanity
# ---------------------------------------------------------------------------
for idx, m in enumerate(run_manifest["models"]):
    try:
        msc.validate_manifest(m)
    except msc.SchemaError as e:
        raise RuntimeError(f"Manifest #{idx} failed validation: {e}")

# ---------------------------------------------------------------------------
# 4)  Persist the collection as **one** JSON artefact
# ---------------------------------------------------------------------------
joined_path = export_dir / f"all_manifests_{run_dtg}.json"
joined_path.write_text(json.dumps(run_manifest, indent=4, ensure_ascii=False), encoding="utf-8")
print("Complete – exported:", joined_path)