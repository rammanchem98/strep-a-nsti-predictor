# Results — SYNTHETIC data

> All numbers below come from a **synthetic** cohort (`data/nsti_gas_SYNTHETIC.csv`) generated to mimic the
> marginals of Katz et al. (BMC Med 2026). They say nothing about real patients and are not comparable
> one-to-one with the paper. Not for clinical use.

## Model selection (validation set, n=82, chosen before the test set was used)

| Model | val balanced acc | val ROC-AUC |
|---|---|---|
| LogReg | 0.725 | 0.834 |
| Random Forest | 0.721 | 0.808 |
| MLP | 0.743 | 0.831 |

Served model: **MLP** (highest validation balanced accuracy).

## Test set (n=82, 25 GAS), used once

Balanced accuracy, precision, recall, F1 at threshold 0.5; sensitivity/specificity at threshold 0.3.
ROC-AUC 95% CI from 1000 bootstrap resamples of the test set.

| Model | ROC-AUC [95% CI] | Bal. acc | Brier | Avg. precision | Precision | Recall | F1 | Sens @0.3 | Spec @0.3 |
|---|---|---|---|---|---|---|---|---|---|
| LogReg | 0.882 [0.785, 0.960] | 0.791 | 0.160 | 0.799 | 0.564 | 0.880 | 0.688 | 0.92 | 0.39 |
| Random Forest | 0.856 [0.749, 0.945] | 0.801 | 0.160 | 0.803 | 0.679 | 0.760 | 0.717 | 0.92 | 0.32 |
| MLP (served) | 0.835 [0.722, 0.928] | 0.815 | 0.168 | 0.726 | 0.636 | 0.840 | 0.724 | 0.92 | 0.47 |
| Oracle (true model) | 0.900 [0.811, 0.966] | 0.771 | 0.117 | 0.834 | 0.933 | 0.560 | 0.700 | 0.84 | 0.84 |
| Paper RF, internal (quoted) | 0.828 [0.763, 0.883] | 0.726 | 0.152 | 0.684 | – | – | – | – | – |
| Paper RF, external (quoted) | 0.727 [0.672, 0.783] | 0.670 | – | – | – | – | – | 0.62* | 0.67* |

\* The paper's threshold-0.3 numbers (Table 4) are from the **external** Dutch validation cohort (n=208),
not internal validation. Paper internal numbers are 1,000-bootstrap estimates on the full cohort (no hold-out);
ours are one 20% hold-out of a synthetic cohort, so CIs are wide.

"Oracle" = the hidden data-generating model's true probabilities: the best achievable performance on this test
set (population ceiling AUC 0.830).

Balanced-accuracy bootstrap CIs: LogReg 0.791 [0.697, 0.872], Random Forest 0.801 [0.704, 0.890], MLP 0.815 [0.717, 0.902]

## Plots

- ROC: `artifacts/roc_SYNTHETIC.png`
- Calibration: `artifacts/calibration_SYNTHETIC.png`
- SHAP (served model): `artifacts/shap_mlp_SYNTHETIC.png`

## SHAP — served model (MLP, KernelExplainer)

Mean |SHAP| on the test set (preprocessed feature space), the sign of the correlation between feature value
and SHAP value, and the true coefficient used by the generator.

| Feature | mean abs SHAP | corr(value, SHAP) | true coef |
|---|---|---|---|
| creatinine_preop | 0.1261 | +0.909 | +1.118 |
| ANOGENITAL_ARRIVAL | 0.0997 | -0.982 | -1.118 |
| SURGERY_BEFORE | 0.0796 | -0.983 | -2.100 |
| UPPER_ARM_ARRIVAL | 0.0492 | +0.982 | +0.895 |
| AGE | 0.0363 | -0.969 | -0.447 |
| LOWER_ARM_ARRIVAL | 0.0302 | +0.983 | +0.224 |
| hgb_preop | 0.0268 | -0.980 | -0.224 |
| DIABETES | 0.0268 | -0.960 | -0.671 |

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
