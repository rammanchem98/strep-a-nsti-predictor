# PROGRESS

## Block 0 — Setup
- Built: git repo (branch `main`, remote `origin` = github.com/rammanchem98/strep-a-nsti-predictor, **not pushed**),
  Python 3.12 venv (`.venv`), pinned `requirements.txt` (CPU torch 2.15, sklearn 1.5.2, shap 0.46, mlflow-skinny 2.19, boruta 0.4.3),
  `src/config.py`, pytest config, `.gitignore` (mlruns, keys, .env).
- Checks: `pytest` 1 passed.
- Deviations: Python 3.12 chosen over system 3.14 (wheel availability for torch/shap).
- Open issues: **GCP project + billing alert NOT done** — needs your gcloud login and creates billable resources. Deferred to Block 6 (will ask).
