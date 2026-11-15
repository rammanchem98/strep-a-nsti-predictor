"""One-shot TEST-set evaluation of the three trained pipelines (SYNTHETIC data).

Writes artifacts/test_metrics_SYNTHETIC.json, plots, SHAP summaries, and results.md; logs test metrics
and plots to each model's MLflow run.
"""
from __future__ import annotations

import json
import warnings

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import shap  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,  # noqa: E402
                             brier_score_loss, f1_score, precision_score, recall_score,
                             roc_auc_score, roc_curve)

from data.generate_synthetic import generate  # noqa: E402
from src import config  # noqa: E402
from src.data_utils import load_splits, split_data, xy  # noqa: E402

warnings.filterwarnings("ignore")
MODELS = ["logreg", "rf", "mlp"]
LABELS = {"logreg": "LogReg", "rf": "Random Forest", "mlp": "MLP", "oracle": "Oracle (true model)"}
N_BOOT = 1000

# Hand-written after the single test run (see PROGRESS.md, block 4); kept here so regeneration is identical.
INTERPRETATION = """
## Interpretation notes

- On this test split the oracle reaches AUC 0.900 (above the 0.830 population ceiling: a favourable 82-patient draw).
  All three models are within their wide, overlapping CIs; LogReg has the highest test AUC, consistent with the
  caveat that a logistic data-generating model favours logistic regression.
- The served MLP was chosen on validation balanced accuracy (0.743 vs 0.725/0.721, ~1.5 patients' difference);
  on the test set it has the highest balanced accuracy but the lowest AUC. Validation differences of this size are noise.
- All three models were trained with class re-weighting (class_weight="balanced" / pos_weight≈2.27), which inflates
  predicted probabilities relative to the 30.8% prevalence (visible in the calibration plot: points below the
  diagonal at mid-range). At the fixed 0.3 flag threshold this gives high sensitivity (0.92) and low specificity
  (0.32–0.47). The threshold was not re-tuned; the paper's 0.3 was chosen on an unweighted RF.
- SHAP signs for the served model match the generator's coefficient signs for all 8 features.
"""


def sens_spec(y, p, t):
    pred = (p >= t).astype(int)
    tp = ((pred == 1) & (y == 1)).sum()
    tn = ((pred == 0) & (y == 0)).sum()
    return tp / (y == 1).sum(), tn / (y == 0).sum()


def bootstrap_ci(y, p, fn, n=N_BOOT, seed=config.SEED):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(fn(y[idx], p[idx]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def metrics(y, p):
    pred = (p >= 0.5).astype(int)
    s3, sp3 = sens_spec(y, p, config.THRESHOLD)
    return {
        "roc_auc": roc_auc_score(y, p),
        "roc_auc_ci": bootstrap_ci(y, p, roc_auc_score),
        "balanced_acc": balanced_accuracy_score(y, pred),
        "balanced_acc_ci": bootstrap_ci(y, p, lambda a, b: balanced_accuracy_score(a, (b >= 0.5).astype(int))),
        "brier": brier_score_loss(y, p),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred),
        "f1": f1_score(y, pred),
        "average_precision": average_precision_score(y, p),
        "sensitivity_at_0.3": s3,
        "specificity_at_0.3": sp3,
    }


def oracle_test_probs(test_y):
    """True hidden-model P(GAS) for the test rows (regenerate + identical split)."""
    df, _ = generate(seed=config.SEED)
    df["_p_true"] = df.attrs["p_true"]
    _, _, test = split_data(df)
    assert (test[config.TARGET].to_numpy() == test_y).all()
    return test["_p_true"].to_numpy()


def shap_values(name, pipe, X_train, X_test):
    pre, clf = pipe.named_steps["pre"], pipe.named_steps["clf"]
    Xt_tr = pre.transform(X_train)[config.FEATURES]
    Xt_te = pre.transform(X_test)[config.FEATURES]
    order = list(pre.get_feature_names_out())
    idx = [order.index(f) for f in config.FEATURES]

    def f(x):  # x in FEATURES order -> model's column order
        z = np.empty_like(x)
        z[:, idx] = x
        return clf.predict_proba(z)[:, 1]

    if name == "rf":
        ex = shap.TreeExplainer(clf)
        sv = ex.shap_values(pre.transform(X_test).to_numpy())
        sv = (sv[:, :, 1] if np.ndim(sv) == 3 else sv[1])[:, idx]
        expl = "TreeExplainer"
    else:
        bg = shap.kmeans(Xt_tr.to_numpy(), 20)
        ex = shap.KernelExplainer(f, bg, seed=config.SEED)
        sv = ex.shap_values(Xt_te.to_numpy(), nsamples=500, silent=True)
        expl = "KernelExplainer"
    return np.asarray(sv), Xt_te, expl


def plot_calibration(y, probs, path):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1, label="perfect")
    for k, p in probs.items():
        fr, mp = calibration_curve(y, p, n_bins=5, strategy="quantile")
        ax.plot(mp, fr, "o-", label=LABELS[k])
    ax.set(xlabel="Predicted P(GAS)", ylabel="Observed fraction GAS",
           title=f"Calibration, test n={len(y)} (SYNTHETIC)", xlim=(0, 1), ylim=(0, 1))
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_roc(y, probs, res, path):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1)
    for k, p in probs.items():
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, label=f"{LABELS[k]} AUC {res[k]['roc_auc']:.3f}")
    ax.set(xlabel="1 - specificity", ylabel="Sensitivity", title="ROC, test set (SYNTHETIC)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def fmt_ci(v, ci):
    return f"{v:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]"


def write_results_md(res, meta, shap_rank, gt, n_test, n_pos):
    served = meta["served_model"]
    P = config.PAPER
    rows = []
    for k in MODELS + ["oracle"]:
        r = res[k]
        tag = " (served)" if k == served else ""
        rows.append(f"| {LABELS[k]}{tag} | {fmt_ci(r['roc_auc'], r['roc_auc_ci'])} | {r['balanced_acc']:.3f} | "
                    f"{r['brier']:.3f} | {r['average_precision']:.3f} | {r['precision']:.3f} | {r['recall']:.3f} | "
                    f"{r['f1']:.3f} | {r['sensitivity_at_0.3']:.2f} | {r['specificity_at_0.3']:.2f} |")
    rows.append(f"| Paper RF, internal (quoted) | {P['internal_roc_auc']:.3f} [{P['internal_roc_auc_ci'][0]:.3f}, "
                f"{P['internal_roc_auc_ci'][1]:.3f}] | {P['internal_balanced_acc']:.3f} | {P['internal_brier']:.3f} | "
                f"{P['internal_pr_auc']:.3f} | – | – | – | – | – |")
    rows.append(f"| Paper RF, external (quoted) | {P['external_roc_auc']:.3f} [{P['external_roc_auc_ci'][0]:.3f}, "
                f"{P['external_roc_auc_ci'][1]:.3f}] | {P['external_balanced_acc']:.3f} | – | – | – | – | – | "
                f"{P['external_sens_at_0.3']:.2f}* | {P['external_spec_at_0.3']:.2f}* |")
    val_rows = "\n".join(f"| {LABELS[k]} | {meta['models'][k]['val_balanced_acc']:.3f} | "
                         f"{meta['models'][k]['val_roc_auc']:.3f} |" for k in MODELS)
    shap_rows = "\n".join(f"| {f} | {v:.4f} | {d:+.3f} | {gt['effective_coefficients'][f]:+.3f} |"
                          for f, v, d in shap_rank)
    md = f"""# Results — SYNTHETIC data

> All numbers below come from a **synthetic** cohort (`data/nsti_gas_SYNTHETIC.csv`) generated to mimic the
> marginals of Katz et al. (BMC Med 2026). They say nothing about real patients and are not comparable
> one-to-one with the paper. Not for clinical use.

## Model selection (validation set, n=82, chosen before the test set was used)

| Model | val balanced acc | val ROC-AUC |
|---|---|---|
{val_rows}

Served model: **{LABELS[served]}** (highest validation balanced accuracy).

## Test set (n={n_test}, {n_pos} GAS), used once

Balanced accuracy, precision, recall, F1 at threshold 0.5; sensitivity/specificity at threshold 0.3.
ROC-AUC 95% CI from {N_BOOT} bootstrap resamples of the test set.

| Model | ROC-AUC [95% CI] | Bal. acc | Brier | Avg. precision | Precision | Recall | F1 | Sens @0.3 | Spec @0.3 |
|---|---|---|---|---|---|---|---|---|---|
{chr(10).join(rows)}

\\* The paper's threshold-0.3 numbers (Table 4) are from the **external** Dutch validation cohort (n=208),
not internal validation. Paper internal numbers are 1,000-bootstrap estimates on the full cohort (no hold-out);
ours are one 20% hold-out of a synthetic cohort, so CIs are wide.

"Oracle" = the hidden data-generating model's true probabilities: the best achievable performance on this test
set (population ceiling AUC {gt['best_achievable_auc_population']:.3f}).

Balanced-accuracy bootstrap CIs: """ + ", ".join(
        f"{LABELS[k]} {fmt_ci(res[k]['balanced_acc'], res[k]['balanced_acc_ci'])}" for k in MODELS) + f"""

## Plots

- ROC: `artifacts/roc_SYNTHETIC.png`
- Calibration: `artifacts/calibration_SYNTHETIC.png`
- SHAP (served model): `artifacts/shap_{served}_SYNTHETIC.png`

## SHAP — served model ({LABELS[served]}, {res[served].get('shap_explainer', '')})

Mean |SHAP| on the test set (preprocessed feature space), the sign of the correlation between feature value
and SHAP value, and the true coefficient used by the generator.

| Feature | mean abs SHAP | corr(value, SHAP) | true coef |
|---|---|---|---|
{shap_rows}
"""
    md += INTERPRETATION
    (config.ROOT / "results.md").write_text(md, encoding="utf-8")


def main():
    train, _, test = load_splits()
    X_tr, _ = xy(train)
    X_te, y = xy(test)
    meta = json.loads(config.SERVED_META.read_text())
    gt = json.loads(config.GROUND_TRUTH.read_text())
    art = config.ARTIFACT_DIR
    art.mkdir(exist_ok=True)

    pipes = {k: joblib.load(config.MODEL_DIR / f"{k}_pipeline.joblib") for k in MODELS}
    probs = {k: pipes[k].predict_proba(X_te)[:, 1] for k in MODELS}
    probs["oracle"] = oracle_test_probs(y)
    res = {k: metrics(y, p) for k, p in probs.items()}

    plot_calibration(y, probs, art / "calibration_SYNTHETIC.png")
    plot_roc(y, probs, res, art / "roc_SYNTHETIC.png")

    shap_summary = {}
    for k in MODELS:
        sv, Xt_te, expl = shap_values(k, pipes[k], X_tr, X_te)
        res[k]["shap_explainer"] = expl
        mean_abs = np.abs(sv).mean(0)
        direction = [float(np.corrcoef(Xt_te.iloc[:, j], sv[:, j])[0, 1]) for j in range(sv.shape[1])]
        shap_summary[k] = sorted(zip(config.FEATURES, mean_abs.tolist(), direction), key=lambda t: -t[1])
        plt.figure()
        shap.summary_plot(sv, Xt_te, show=False, plot_size=(7, 4))
        plt.title(f"SHAP — {LABELS[k]} (SYNTHETIC test set)")
        plt.tight_layout()
        plt.savefig(art / f"shap_{k}_SYNTHETIC.png", dpi=120)
        plt.close("all")

    out = {"data": "SYNTHETIC", "n_test": int(len(y)), "n_pos": int(y.sum()), "served_model": meta["served_model"],
           "metrics": res, "shap": shap_summary}
    (art / "test_metrics_SYNTHETIC.json").write_text(json.dumps(out, indent=2, default=float))
    write_results_md(res, meta, shap_summary[meta["served_model"]], gt, len(y), int(y.sum()))

    mlflow.set_tracking_uri(config.MLRUNS_DIR.as_uri())
    for k in MODELS:
        with mlflow.start_run(run_id=meta["models"][k]["run_id"]):
            mlflow.log_metrics({f"test_{m}": v for m, v in res[k].items() if isinstance(v, float)})
            mlflow.log_metrics({"test_roc_auc_ci_low": res[k]["roc_auc_ci"][0],
                                "test_roc_auc_ci_high": res[k]["roc_auc_ci"][1]})
            for f in ["calibration_SYNTHETIC.png", "roc_SYNTHETIC.png", f"shap_{k}_SYNTHETIC.png"]:
                mlflow.log_artifact(str(art / f))
    for k in MODELS + ["oracle"]:
        r = res[k]
        print(f"{k:7s} AUC {fmt_ci(r['roc_auc'], r['roc_auc_ci'])} balacc {r['balanced_acc']:.3f} "
              f"brier {r['brier']:.3f} AP {r['average_precision']:.3f} "
              f"sens/spec@0.3 {r['sensitivity_at_0.3']:.2f}/{r['specificity_at_0.3']:.2f}")


if __name__ == "__main__":
    main()
