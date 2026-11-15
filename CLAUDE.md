# CLAUDE.md — nsti-gas-mlops (one-day build)

## How to work (autonomous mode)
- Build Blocks 0–7 in order without waiting for confirmation between blocks. No teaching explanations needed.
- Each block must end with its "done when" check passing, then: `pytest` green, one git commit
  (`block N: <summary>`), and an entry appended to `PROGRESS.md`:
  block, what was built, check results (numbers), deviations from this file, open issues.
- I will review PROGRESS.md and commits between blocks. Keep entries short and factual.
- If a check fails, try to fix it up to 3 times. If still failing, record it in PROGRESS.md and
  continue only if later blocks don't depend on it; otherwise stop and tell me.
- STOP and ask me before: anything that needs my credentials or login (gcloud auth, GitHub secrets,
  Workload Identity setup), anything that costs money or creates cloud resources, pushing to a
  remote, deleting files outside this repo, or changing a decision written in this file.
  Do all local work up to that point first.
- Never weaken a check or change a metric/threshold to make a result look better. Report weak results honestly.
- Fixed random seeds everywhere (seed=42) so runs are reproducible.

## The paper (reference, not reproduced exactly)
Katz S, Suijker J, Skrede S, et al. "A validated model for early prediction of group A
streptococcal aetiology in necrotising soft tissue infections using minimal patient data."
BMC Medicine 2026;24:76. doi:10.1186/s12916-025-04593-y
Authors' code (MIT): https://github.com/sonjakatz/permit-nsti-gas
- Use their code, grid and summary CSVs as reference ONLY.
- Do NOT load their .pickle/.sav files (SHAP/predProba pickles may hold real patient-level values).

Paper results (Random Forest, pre-surgery model):
- Internal (1,000 bootstrap): ROC-AUC 0.828 [0.763, 0.883], balanced acc 0.726, Brier 0.152, PR-AUC 0.684
- External (Dutch, n=208): ROC-AUC 0.727 [0.672, 0.783], balanced acc 0.670; after fine-tuning on 30%: 0.758
- Threshold 0.3 (chosen to minimise false negatives, Table 4, EXTERNAL model): sensitivity 0.62, specificity 0.67

## Data decision: SYNTHETIC
Real INFECT data is not public (on request from E. Saccenti / A. Norrby-Teglund). All data here is
synthetic. Label it SYNTHETIC in file names, MLflow tags, API responses and README.

### Features (exact names from the authors' Boruta output)
AGE, DIABETES, UPPER_ARM_ARRIVAL, LOWER_ARM_ARRIVAL, ANOGENITAL_ARRIVAL, SURGERY_BEFORE,
creatinine_preop, hgb_preop. Target: GAS (0/1).

### Marginals to match (paper Table 3, development cohort n=409)
| Feature | Value |
|---|---|
| GAS | 126 (30.8%) |
| AGE | mean 58.7, SD 15.1 (clip 18–100) |
| UPPER_ARM / LOWER_ARM / ANOGENITAL | 11.7% / 13.7% / 35.0% (mutually exclusive; rest = other sites) |
| SURGERY_BEFORE | 18.8% |
| DIABETES | 24.0% |
| creatinine_preop | mean 161.2, SD 121.3 µmol/L → lognormal, sigma≈0.67, mu≈4.86 (median ≈129) |
| hgb_preop | mean 11.4, SD 3.04 (assume g/dL), clip to plausible range |

The paper gives NO GAS vs non-GAS split. Class signal is our assumption, from SHAP directions:
- Upper arm → toward GAS. Anogenital, diabetes, recent surgery, age > 50 → away from GAS.
- Creatinine > ~110 µmol/L is the strongest pointer toward GAS.
- One anchored effect: 5/126 GAS had recent surgery vs 72/283 non-GAS → OR ≈ 0.12, coef ≈ −2.1.

### Generator design (two layers)
1. Draw features to match the marginals; mild correlation (older → more diabetes).
2. Hidden logistic model for P(GAS). Signs from SHAP; intercept tuned so GAS rate = 30.8%;
   overall strength tuned so the best achievable (true-model) AUC ≈ 0.83.
3. Add 3 pure-noise decoy columns (for testing Boruta).
4. Inject 5–15% missing values in creatinine_preop and hgb_preop.
5. Save `ground_truth.json` (true coefficients, seed, best achievable AUC).
Caveat to state in the README: a logistic true model favours LogReg. Optional variant with a creatinine
threshold effect at 110 to make the comparison fairer.

## Method decisions (and where we differ from the paper)
- Split: stratified 60/20/20 train/val/test. (Paper: LOOCV + bootstrap on the full set, no hold-out.)
  ~82 test patients → wide CIs; report bootstrap CIs. Optional: repeat over 5 seeds.
- Preprocessing (authors' code): numeric → MinMaxScaler + IterativeImputer(sample_posterior=True,
  max_iter=10, n_nearest_features=5); binary → KNNImputer(k=5). Fit on train only.
- Boruta (on train only): 80% subsample per iteration, 30 iterations, keep if selected in >50%.
  (Paper ran Boruta 50× per iteration; we run once per iteration to save time.)
  Check: keeps the 8 real features, rejects the decoys.
- Models (comparison kept):
  - Logistic regression baseline
  - Random Forest, grid from the authors' code: n_estimators [100,300,700], max_depth [2,4,6],
    max_features [2,4,6]; scored on balanced accuracy
  - PyTorch MLP 8→16→8→1, BCEWithLogitsLoss(pos_weight≈2.25), dropout, weight decay,
    early stopping on val; small grid over width / lr / dropout / weight decay
- Choose the served model by VALIDATION balanced accuracy, recorded before touching the test set.
- Test set used once.

## Metrics (test set)
ROC-AUC with 1,000-sample bootstrap 95% CI, balanced accuracy, Brier, precision, recall, F1,
average precision, sensitivity and specificity at threshold 0.3. Calibration plot. SHAP for the
served model (TreeExplainer for RF; KernelExplainer for MLP/LogReg).
Results table rows: LogReg | RF | MLP | Paper RF (quoted). State that the paper's 0.3 numbers are external.

## MLOps scope (one day)
- MLflow: local file store (`mlruns/`), log params, metrics, plots, model. No Postgres server today.
- Serving: one sklearn Pipeline (imputation + scaling + model) logged as a single artefact. No separate scaler pickle.
- FastAPI `/predict`: 8 fields with Pydantic range validation → probability, flag at 0.3,
  "synthetic / not for clinical use" disclaimer. `/health` endpoint.
- pytest: data (marginals, GAS rate, split stratification), models (fit/predict shapes), API (TestClient).
- Docker: single slim image.
- Cloud: Google Cloud Run (scale to zero), image in Artifact Registry. Set a billing alert.
- GitHub Actions: pytest → build → deploy, Workload Identity Federation (no JSON keys). Images tagged with git SHA.
- No secrets in git. No Kubernetes/EKS today.

## Blocks (≈10h)
0. Setup: repo, venv, GCP project, billing alert (30m)
1. Synthetic generator + marginal checks (1.5h)
2. Splits + Boruta (1h)
3. Models + MLflow (2h)
4. Evaluation + SHAP + results.md (1.5h)
5. API + tests + Docker (1.5h)
6. Cloud Run + CI/CD (1.5h)
7. README (45m)
If behind, cut in order: manual `gcloud run deploy` → SHAP for served model only → skip Boruta (use the paper's 8).

## Layout
```
data/generate_synthetic.py
src/ config.py data_utils.py features.py models.py train.py evaluate.py api.py
tests/ test_data.py test_models.py test_api.py
Dockerfile  .github/workflows/ci-cd.yaml  README.md  results.md
```

## README must include
Paper citation and link; SYNTHETIC DATA banner; "learning/portfolio project, not clinically validated,
must not be used for clinical decisions"; results table vs paper; every difference from the paper and why;
live Cloud Run link.
