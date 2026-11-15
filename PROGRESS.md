# PROGRESS

## Block 0 — Setup
- Built: git repo (branch `main`, remote `origin` = github.com/rammanchem98/strep-a-nsti-predictor, **not pushed**),
  Python 3.12 venv (`.venv`), pinned `requirements.txt` (CPU torch 2.15, sklearn 1.5.2, shap 0.46, mlflow-skinny 2.19, boruta 0.4.3),
  `src/config.py`, pytest config, `.gitignore` (mlruns, keys, .env).
- Checks: `pytest` 1 passed.
- Deviations: Python 3.12 chosen over system 3.14 (wheel availability for torch/shap).
- Open issues: **GCP project + billing alert NOT done** — needs your gcloud login and creates billable resources. Deferred to Block 6 (will ask).

## Block 1 — Synthetic generator
- Built: `data/generate_synthetic.py` → `data/nsti_gas_SYNTHETIC.csv` (n=409) + `data/ground_truth.json`;
  `--variant threshold` → `*_threshold.csv/json` (creatinine step at 110). `src/data_utils.py`, `tests/test_data.py`.
- Method: hidden logistic model; strength multiplier (1.118) + intercept tuned on a 200k population so
  GAS rate = 30.8% and true-model AUC = 0.830. In the 409 sample, binary counts are allocated exactly
  (GAS = top-126 of η + logistic noise, i.e. intercept re-solved in-sample) and normal draws are standardised.
- Effective coefs: UPPER +0.89, LOWER +0.22, ANOGENITAL −1.12, DIABETES −0.67, AGE[(x−50)/15] −0.45,
  creat[log(c/110)/0.67] +1.12, hgb[z] −0.22, SURGERY −2.1 (anchored, unscaled).
- Checks: GAS 126/409 (30.8%); UPPER/LOWER/ANOG 11.7/13.7/35.0%; SURG 18.8%; DIAB 24.0%;
  AGE 58.7±15.1; creat mean 158.5 SD 112.9 median 130.7; hgb 11.4±3.05; missing creat 10.0%, hgb 8.1%.
  Best achievable AUC: population 0.830, this sample 0.867. `pytest` 18 passed.
- Deviations: exact-count allocation in the sample (not in CLAUDE.md, keeps marginals = Table 3).
  Lower-arm and hgb directions not given by paper → small assumed weights (+0.2, −0.2).
- Open issues: surgery split is 10/126 GAS vs 67/283 non-GAS (marginal OR ≈ 0.28) vs paper 5/72 (OR ≈ 0.12);
  coefficient kept at the anchored −2.1 (conditional OR ≠ marginal OR + sampling noise). Sample true AUC (0.867)
  is above the 0.83 target by chance; models will be compared against this ceiling.

## Block 2 — Splits + Boruta
- Built: `src/features.py` (authors' preprocessor as a ColumnTransformer; Boruta loop), splits saved to
  `data/splits/{train,val,test}_SYNTHETIC.csv` (245/82/82; GAS 75/26/25, stratified),
  `artifacts/boruta_SYNTHETIC.csv`, `artifacts/boruta_selected.json`, `tests/test_features.py`.
- Boruta config = CLAUDE.md + authors' script (RF max_depth=5, balanced, perc=100, max_iter=100), on train only,
  30 × 80% subsamples (196 rows), keep if confirmed in >50%.
- **Check FAILED**: selected = [creatinine_preop] only (1.00). Others: hgb 0.03, AGE 0.03, NOISE_2 0.03, rest 0.00.
  Decoys rejected ✔; 7/8 real features rejected ✘.
- Diagnosis (not a code bug; settings match the authors'): low power at n≈196 + impurity-importance bias toward
  continuous columns (continuous shadow features beat binary effects). Diagnostic runs (10 iters, NOT used for selection):
  full 409 rows → creat 1.0, AGE 0.3, rest 0; fresh synthetic n=1500 → AGE/SURGERY/creat/hgb 1.0, ANOG 0.8,
  NOISE_1 0.5 (!), DIABETES/UPPER/LOWER 0.0.
- Deviation: models in Blocks 3+ use the **paper's 8 features** (CLAUDE.md's documented fallback), not the Boruta output.
  Boruta settings were not tuned to force a pass.
- Open issues: whether to keep the Boruta result as-is in README (planned: yes, reported as a negative result).

## Block 3 — Models + MLflow
- Built: `src/models.py` (LogReg + RF GridSearchCV pipelines; `TorchMLPClassifier` sklearn wrapper, picklable),
  `src/train.py`, `tests/test_models.py`. MLflow file store `mlruns/` (gitignored), experiment `nsti-gas-SYNTHETIC`,
  tag `data=SYNTHETIC`; params, val metrics, model (with input example), MLP grid JSON. Served run tagged `served=true`.
  Pipelines saved to `model/*.joblib`; `model/served_pipeline.joblib` + `served_meta.json` (selection recorded pre-test).
- Hyperparameter selection: LogReg (penalty None/l2 × C 1e-3..1e3) and RF (authors' 27-point grid) by 5-fold CV
  balanced acc on train; MLP grid (hidden (16,8)/(32,16) × lr 1e-3/3e-3 × dropout 0.1/0.3 × wd 1e-4/1e-3), full-batch
  AdamW, early stopping on val loss (patience 100), pos_weight = 170/75 = 2.27.
- Results (validation, n=82, threshold 0.5):
  | model | best params | CV bal acc | val bal acc | val AUC | val Brier |
  |---|---|---|---|---|---|
  | LogReg | l2, C=100 | 0.765 | 0.725 | 0.834 | 0.183 |
  | RF | depth 4, max_feat 2, 300 trees | 0.749 | 0.721 | 0.808 | 0.181 |
  | MLP | (32,16), lr 1e-3, dropout 0.3, wd 1e-4 | – | **0.743** | 0.831 | 0.185 |
- Served model: **MLP** (max val balanced acc). Re-run from clean reproduces identical numbers. `pytest` 26 passed.
- Deviations: `class_weight="balanced"` on LogReg/RF (authors used none) so all three models handle imbalance the
  same way as the MLP's pos_weight — otherwise balanced acc at 0.5 is not comparable. LogReg grid taken from authors' code.
- Open issues: the val-set gap (0.743 vs 0.725) is ~1.5 patients' worth and the MLP was grid-searched + early-stopped on
  the same val set, so its val score is optimistically biased. Selection rule followed as written. Serving MLP means
  the Docker image needs CPU torch.

## Block 4 — Evaluation + SHAP + results.md
- Built: `src/evaluate.py` → `results.md`, `artifacts/test_metrics_SYNTHETIC.json`, ROC + calibration plots,
  SHAP summary plots for all 3 models (TreeExplainer RF; KernelExplainer LogReg/MLP, k-means(20) background,
  preprocessed space). Test metrics + plots logged to each MLflow run. Oracle row = generator's true P(GAS).
- Test set (n=82, 25 GAS):
  | model | AUC [95% CI] | bal acc | Brier | AP | sens/spec @0.3 |
  |---|---|---|---|---|---|
  | LogReg | 0.882 [0.785, 0.960] | 0.791 | 0.160 | 0.799 | 0.92 / 0.39 |
  | RF | 0.856 [0.749, 0.945] | 0.801 | 0.160 | 0.803 | 0.92 / 0.32 |
  | MLP (served) | 0.835 [0.722, 0.928] | 0.815 | 0.168 | 0.726 | 0.92 / 0.47 |
  | Oracle | 0.900 [0.811, 0.966] | 0.771 | 0.117 | 0.834 | 0.84 / 0.84 |
- SHAP (served MLP): sign matches the true coefficient for 8/8 features; top 3 = creatinine, anogenital, surgery.
- `pytest` 26 passed.
- Deviations / honesty notes:
  - `evaluate` was executed twice on the test set (2nd run only to inspect a stderr traceback = loky `wmic` core-count
    warning on Windows, now silenced via `LOKY_MAX_CPU_COUNT`). Outputs identical; no model/threshold/selection change.
  - An interpretation section was hand-written into results.md after the run (calibration over-prediction from class
    re-weighting → low specificity at 0.3; LogReg best AUC as expected under a logistic truth). Stored in
    `evaluate.INTERPRETATION`; re-rendering from the saved JSON reproduces results.md byte-for-byte.
  - Optional 5-seed repeat not done.
- Open issues: served model (MLP) has the lowest test AUC of the three; selection rule followed as written.
  Low specificity at the 0.3 flag is reported, not re-tuned.

## Block 5 — API + tests + Docker
- Built: `src/api.py` (FastAPI; `/health`, `/predict`), `tests/test_api.py`, `Dockerfile` (python:3.12-slim,
  non-root, `$PORT` default 8080), `.dockerignore`, `requirements-serve.txt` (no mlflow/shap/boruta in the image).
- `/predict`: 8 fields, Pydantic ranges (AGE 18–100 int; binaries Literal[0,1]; creatinine 20–2000 µmol/L and
  hgb 3–22 g/dL, both optional → imputed by the pipeline); arrival sites mutually exclusive; extra fields rejected.
  Returns probability, `gas_flag` (p ≥ 0.3), threshold, model, `data: SYNTHETIC`, disclaimer.
- Checks: `pytest` 38 passed (1 third-party starlette/anyio DeprecationWarning). Docker build OK; container smoke test:
  `/health` 200, `/predict` (upper arm, creat 180, hgb null) → p=0.868 flag=true, invalid body → 422.
- Deviations: none from CLAUDE.md. Image is 1.91 GB (CPU torch ≈ 810 MB, no CUDA libs) because the served model is the MLP.
- Open issues: if image size/cold-start matters on Cloud Run, the alternative is serving LogReg/RF (sklearn-only,
  ~400 MB) — that would change the CLAUDE.md selection rule, so not done.

## Block 6 — Cloud Run + CI/CD (LOCAL PREP ONLY — NOT DEPLOYED)
- Built: `.github/workflows/ci-cd.yaml` (pytest → docker build → push to Artifact Registry tagged `${{ github.sha }}`
  → `deploy-cloudrun` with min-instances 0, max 2, 1 GiB, unauthenticated, label data=synthetic → `/health` smoke test;
  WIF auth via `id-token: write`, no JSON keys; config in repo *variables*, not secrets).
  `deploy/gcp_setup.sh` (project + billing link, APIs, **budget alert** 50/90/100% of $10, Artifact Registry repo `nsti`,
  deployer + no-role runtime service accounts, WIF pool/provider restricted to `rammanchem98/strep-a-nsti-predictor`).
- Checks: workflow YAML parses (jobs: test, build-deploy); `bash -n` OK on setup script. Nothing executed against GCP or GitHub.
- **STOPPED here per CLAUDE.md**: needs your gcloud login, billing account, creates billable resources, sets GitHub
  variables, and a push. gcloud already has 2 accounts and an active project configured locally — not used, awaiting your choice.
- Open issues: region default `europe-west4` (assumption — change if you prefer); live Cloud Run URL pending.

## Block 7 — README
- Built: `README.md` — SYNTHETIC banner, not-for-clinical-use statement, paper citation + DOI + authors' repo,
  results table vs paper (internal + external rows, 0.3 numbers flagged as external), synthetic data description,
  logistic-truth caveat, full "differences from the paper and why" table, Boruta negative result, architecture,
  local/Docker/deploy instructions.
- Checks: `pytest` 38 passed.
- Open issues: **live Cloud Run link is a placeholder** until Block 6 deployment is done.
