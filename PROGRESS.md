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
