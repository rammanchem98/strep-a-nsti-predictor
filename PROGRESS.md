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
