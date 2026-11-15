# NSTI Group A Strep predictor — MLOps learning project

> ## ⚠️ SYNTHETIC DATA
> Every patient record in this repository is **synthetic**, generated to mimic published summary statistics.
> No real patient data is used. **This is a learning/portfolio project. It is not clinically validated and
> must not be used for clinical decisions.**

**Live API:** _pending deployment_ (Cloud Run URL will be added here)

An end-to-end MLOps rebuild, on synthetic data, of the pre-surgery model from:

> Katz S, Suijker J, Skrede S, et al. *A validated model for early prediction of group A streptococcal aetiology
> in necrotising soft tissue infections using minimal patient data.* BMC Medicine 2026;24:76.
> [doi:10.1186/s12916-025-04593-y](https://doi.org/10.1186/s12916-025-04593-y) ·
> authors' code (MIT): [sonjakatz/permit-nsti-gas](https://github.com/sonjakatz/permit-nsti-gas)

The paper predicts whether a necrotising soft tissue infection (NSTI) is caused by group A *Streptococcus* (GAS)
using 8 variables available before surgery. The real INFECT cohort is not public, so this project generates a
synthetic cohort, trains three models, tracks them in MLflow, and serves the selected one as a FastAPI service
on Google Cloud Run with GitHub Actions CI/CD.

## Results (synthetic test set, n=82, 25 GAS)

| Model | ROC-AUC [95% CI] | Bal. acc | Brier | Avg. precision | Sens @0.3 | Spec @0.3 |
|---|---|---|---|---|---|---|
| Logistic regression | 0.882 [0.785, 0.960] | 0.791 | 0.160 | 0.799 | 0.92 | 0.39 |
| Random forest | 0.856 [0.749, 0.945] | 0.801 | 0.160 | 0.803 | 0.92 | 0.32 |
| **MLP (served)** | 0.835 [0.722, 0.928] | 0.815 | 0.168 | 0.726 | 0.92 | 0.47 |
| Oracle (true generator) | 0.900 [0.811, 0.966] | 0.771 | 0.117 | 0.834 | 0.84 | 0.84 |
| Paper RF, internal (quoted) | 0.828 [0.763, 0.883] | 0.726 | 0.152 | 0.684 | – | – |
| Paper RF, external (quoted) | 0.727 [0.672, 0.783] | 0.670 | – | – | 0.62* | 0.67* |

\* The paper's threshold-0.3 sensitivity/specificity are from the **external** Dutch cohort (Table 4), not internal validation.

**How to read this:** the synthetic numbers are *not* evidence about real patients, and they are not comparable
one-to-one with the paper. The generator was tuned so the best achievable AUC is ≈0.83 in the population, so
matching the paper's AUC is by construction. On this particular 82-patient test split even the oracle scores 0.90.
All three models' CIs overlap heavily. Full details, calibration and SHAP: [results.md](results.md).

Key findings:
- The MLP was served because it had the best **validation** balanced accuracy (0.743 vs 0.725 / 0.721). On the
  test set it has the best balanced accuracy but the lowest AUC. Differences of this size are noise at n=82.
- Logistic regression has the best test AUC, as expected: the hidden data-generating model is logistic (see caveats).
- Class re-weighting inflates predicted probabilities, so the fixed 0.3 flag gives high sensitivity (0.92) and low
  specificity (0.32–0.47). The threshold was kept at the paper's value, not re-tuned.
- SHAP for the served model recovers the generator's direction for all 8 features
  (creatinine ↑, upper arm ↑, anogenital ↓, recent surgery ↓, diabetes ↓, age ↓).
- **Boruta did not recover the feature set** (see below).

## Synthetic data

`data/generate_synthetic.py` (seed 42) builds n=409 patients matching the paper's Table 3 marginals:
GAS 126 (30.8%); age 58.7 ± 15.1; upper arm / lower arm / anogenital 11.7 / 13.7 / 35.0% (mutually exclusive);
recent surgery 18.8%; diabetes 24.0%; creatinine lognormal (mean ≈ 161, median ≈ 129 µmol/L); Hb 11.4 ± 3.0 g/dL;
10% / 8% missing creatinine / Hb; plus 3 pure-noise decoy columns.

The paper reports **no GAS vs non-GAS split**, so the label model is our assumption. It is a hidden logistic model
whose signs come from the paper's SHAP plots. Recent surgery is anchored at coefficient −2.1, from the paper's 5/126 vs
72/283 counts. The intercept and an overall strength factor are tuned so the GAS rate is 30.8% and the true-model AUC is
0.83. The true coefficients are in [data/ground_truth.json](data/ground_truth.json).
`--variant threshold` writes an alternative cohort with a creatinine step effect at 110 µmol/L. It is generated but
not used for the results above.

**Caveat:** a logistic ground truth favours logistic regression. The model comparison is about pipeline mechanics,
not about which algorithm is better for real NSTI data.

## Differences from the paper, and why

| Topic | Paper | Here | Why |
|---|---|---|---|
| Data | Real INFECT cohort (n=409) + Dutch external cohort (n=208) | Synthetic n=409, no external cohort | Real data not public |
| Validation | LOOCV + 1,000 bootstrap on full data, no hold-out | Stratified 60/20/20 train/val/test, test used once, 1,000-bootstrap CIs on test | Clean model selection + one unbiased estimate for an MLOps demo; costs width of CIs (82 test patients) |
| Feature selection | Boruta, 50 runs × bootstrap iterations → 8 features | Boruta 30 × 80% subsamples of train, 1 run each; **models use the paper's 8 features** | Time; Boruta here selected only creatinine (below) |
| Models | RF (and others compared) | LogReg, RF (authors' grid), PyTorch MLP | Comparison requested; MLP to exercise a non-sklearn model inside an sklearn Pipeline |
| Class imbalance | No weighting in authors' code | `class_weight="balanced"` (LogReg, RF), `pos_weight≈2.27` (MLP) | Same imbalance handling for all three, so validation balanced accuracy is a fair selection criterion |
| Served model | RF | Whatever wins validation balanced accuracy (MLP) | Pre-registered rule, recorded before test evaluation |
| Hyperparameters | GridSearchCV, cv=5 | Same for LogReg/RF on train; MLP grid with early stopping on val | MLP needs a val set for early stopping |
| Preprocessing | MinMaxScaler + IterativeImputer (numeric), KNNImputer (binary) | Same, fit on train only, inside one serialised Pipeline | Avoid leakage; one artefact to serve |
| Threshold 0.3 | Chosen to minimise false negatives on the **external** model | Applied unchanged | Kept for comparability; not re-tuned on our test set |

### Boruta: negative result
With the paper's settings (RF max_depth 5, perc 100, balanced) on 196-row subsamples, Boruta confirmed **only
creatinine** (100% of runs). All other real features scored 0–3%, and all 3 decoys were rejected. Diagnostic runs show
this is a power problem plus the impurity-importance bias toward continuous columns. At n=1,500 Boruta confirms age,
surgery, creatinine, Hb and anogenital, still misses diabetes and both arm sites, and gives a continuous noise column
50%. The settings were not tuned to force a pass. Models use the paper's 8 features instead.

## Architecture

```
data/generate_synthetic.py  ─► data/nsti_gas_SYNTHETIC.csv, ground_truth.json
src/features.py             ─► splits (60/20/20), preprocessing, Boruta
src/train.py                ─► LogReg / RF / MLP pipelines ─► MLflow (mlruns/, tag data=SYNTHETIC)
                                └► model/served_pipeline.joblib  (imputer + scaler + model, one artefact)
src/evaluate.py             ─► results.md, artifacts/ (ROC, calibration, SHAP), test metrics → MLflow
src/api.py (FastAPI)        ─► /health, /predict  ─► Docker ─► Artifact Registry ─► Cloud Run (scale to zero)
.github/workflows/ci-cd.yaml: pytest → build (tag = git SHA) → deploy, auth via Workload Identity Federation
```

## Run locally

```bash
py -3.12 -m venv .venv && .venv/Scripts/activate        # Linux/macOS: python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python data/generate_synthetic.py      # synthetic cohort + ground truth
python -m src.features                 # splits + Boruta (~4 min)
python -m src.train                    # models + MLflow + served model selection
python -m src.evaluate                 # test-set evaluation (run once) → results.md
pytest                                 # 38 tests: data, features, models, API
mlflow ui --backend-store-uri ./mlruns # needs full `mlflow`; requirements pin mlflow-skinny
uvicorn src.api:app --port 8080
```

Docker:

```bash
docker build -t nsti-gas-api .
docker run -p 8080:8080 nsti-gas-api
curl -X POST localhost:8080/predict -H "Content-Type: application/json" -d '{"AGE":45,"DIABETES":0,"UPPER_ARM_ARRIVAL":1,"LOWER_ARM_ARRIVAL":0,"ANOGENITAL_ARRIVAL":0,"SURGERY_BEFORE":0,"creatinine_preop":180,"hgb_preop":12.1}'
```

Response:
`{"probability_gas": …, "gas_flag": true|false, "threshold": 0.3, "model": "mlp", "data": "SYNTHETIC", "disclaimer": "…"}`.
`creatinine_preop` (µmol/L) and `hgb_preop` (g/dL) may be `null` and are then imputed. At most one arrival site may be 1.

## Deploy (Cloud Run)

One-time setup: review and run [deploy/gcp_setup.sh](deploy/gcp_setup.sh). It creates the project link, a budget
alert, Artifact Registry, service accounts and Workload Identity Federation restricted to this repo. Then set the
printed GitHub repository variables (`GCP_PROJECT_ID`, `GCP_REGION`, `GCP_WIF_PROVIDER`, `GCP_DEPLOY_SA`,
`GCP_RUNTIME_SA`). After that, every push to `main` runs tests, builds an image tagged with the git SHA, and
deploys it. No service-account keys are stored anywhere.

## Licence and attribution

Method and feature names follow Katz et al. (2026) and the authors' MIT-licensed code. That code was used as a
reference only: none of their data files or pickles were loaded. Build log: [PROGRESS.md](PROGRESS.md).
