"""Train LogReg / RF / MLP on the SYNTHETIC train split, select the served model on VALIDATION
balanced accuracy, and log everything to a local MLflow file store. The test split is not touched here.
"""
from __future__ import annotations

import json
import os
import warnings

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, roc_auc_score

from src import config
from src.data_utils import load_splits, xy
from src.models import logreg_search, mlp_search, rf_search

warnings.filterwarnings("ignore", category=UserWarning)
os.environ.setdefault("PYTHONWARNINGS", "ignore::UserWarning")  # also silences joblib workers


def val_metrics(pipe, X_val, y_val):
    p = pipe.predict_proba(X_val)[:, 1]
    return {"val_balanced_acc": balanced_accuracy_score(y_val, (p >= 0.5).astype(int)),
            "val_roc_auc": roc_auc_score(y_val, p), "val_brier": brier_score_loss(y_val, p)}


def setup_mlflow():
    mlflow.set_tracking_uri(config.MLRUNS_DIR.as_uri())
    mlflow.set_experiment(config.EXPERIMENT)


def main():
    np.random.seed(config.SEED)
    setup_mlflow()
    train, val, _ = load_splits()
    X_tr, y_tr = xy(train)
    X_val, y_val = xy(val)
    config.MODEL_DIR.mkdir(exist_ok=True)
    tags = {"data": "SYNTHETIC", "seed": str(config.SEED), "features": ",".join(config.FEATURES)}
    summary = {}

    for name, search_fn in [("logreg", logreg_search), ("rf", rf_search)]:
        with mlflow.start_run(run_name=name, tags=tags) as run:
            gs = search_fn().fit(X_tr, y_tr)
            pipe = gs.best_estimator_
            m = val_metrics(pipe, X_val, y_val) | {"cv_balanced_acc": gs.best_score_}
            mlflow.log_params({k.replace("clf__", ""): v for k, v in gs.best_params_.items()})
            mlflow.log_metrics(m)
            mlflow.sklearn.log_model(pipe, artifact_path="model", input_example=X_tr.head(3))
            joblib.dump(pipe, config.MODEL_DIR / f"{name}_pipeline.joblib")
            summary[name] = {"run_id": run.info.run_id, "params": gs.best_params_, **m}
            print(name, gs.best_params_, {k: round(v, 3) for k, v in m.items()})

    with mlflow.start_run(run_name="mlp", tags=tags) as run:
        pipe, params, grid = mlp_search(X_tr, y_tr, X_val, y_val)
        m = val_metrics(pipe, X_val, y_val)
        mlflow.log_params({**{k: str(v) for k, v in params.items()},
                           "pos_weight": round(pipe.named_steps["clf"].pos_weight_, 3),
                           "best_epoch": pipe.named_steps["clf"].best_epoch_})
        mlflow.log_metrics(m)
        mlflow.log_dict({"grid": [{"params": {k: str(v) for k, v in p.items()}, "val_bal_acc": s,
                                   "best_epoch": e} for p, s, e in grid]}, "mlp_grid.json")
        mlflow.sklearn.log_model(pipe, artifact_path="model", input_example=X_tr.head(3))
        joblib.dump(pipe, config.MODEL_DIR / "mlp_pipeline.joblib")
        summary["mlp"] = {"run_id": run.info.run_id, "params": {k: str(v) for k, v in params.items()}, **m}
        print("mlp", params, {k: round(v, 3) for k, v in m.items()})

    # Served model chosen on validation balanced accuracy, recorded BEFORE any test evaluation.
    chosen = max(summary, key=lambda k: summary[k]["val_balanced_acc"])
    joblib.dump(joblib.load(config.MODEL_DIR / f"{chosen}_pipeline.joblib"), config.SERVED_MODEL)
    meta = {"served_model": chosen, "selection": "max validation balanced accuracy (threshold 0.5)",
            "threshold_flag": config.THRESHOLD, "data": "SYNTHETIC", "features": config.FEATURES,
            "models": summary}
    config.SERVED_META.write_text(json.dumps(meta, indent=2, default=str))
    with mlflow.start_run(run_id=summary[chosen]["run_id"]):
        mlflow.set_tag("served", "true")
    print("served model:", chosen)


if __name__ == "__main__":
    main()
