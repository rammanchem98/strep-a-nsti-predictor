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
