"""Generate a SYNTHETIC NSTI cohort that mimics the marginals of Katz et al. (BMC Med 2026, Table 3).

Two layers:
  1. Features drawn to match published marginals (n=409 development cohort).
  2. A hidden logistic model for P(GAS). Directions come from the paper's SHAP plots; magnitudes
     are OUR ASSUMPTION (the paper gives no GAS vs non-GAS split). Intercept is tuned so the
     population GAS rate is 30.8%; a global strength multiplier is tuned so the true-model
     (best achievable) population ROC-AUC is ~0.83.

Usage:
    python data/generate_synthetic.py                    # logistic truth (default)
    python data/generate_synthetic.py --variant threshold  # creatinine step effect at 110 umol/L

No real patient data is used or loaded anywhere.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402

# --- Published marginals (Table 3) --------------------------------------------------------
GAS_RATE = 126 / 409          # 0.308
AGE_MEAN, AGE_SD = 58.7, 15.1
P_UPPER, P_LOWER, P_ANOG = 0.117, 0.137, 0.350
P_SURGERY = 0.188
P_DIABETES = 0.240
CREAT_MU, CREAT_SIGMA = 4.86, 0.67   # lognormal -> mean ~161, median ~129 umol/L
HGB_MEAN, HGB_SD = 11.4, 3.04        # g/dL (assumed unit)
MISSING = {"creatinine_preop": 0.10, "hgb_preop": 0.08}  # within the 5-15% band

# --- Hidden-model weights (directions from paper SHAP; magnitudes assumed) ----------------
# Applied to: binaries as 0/1, AGE as (AGE-50)/15, creatinine as log(creat/110)/0.67,
# hgb as (hgb-11.4)/3.04. These are multiplied by a tuned strength factor; SURGERY_BEFORE is
# anchored at -2.1 (paper counts 5/126 vs 72/283 -> OR ~0.12) and NOT scaled.
BASE_WEIGHTS = {
    "UPPER_ARM_ARRIVAL": 0.8,     # toward GAS
    "LOWER_ARM_ARRIVAL": 0.2,     # weak, direction unclear in paper
    "ANOGENITAL_ARRIVAL": -1.0,   # away from GAS
    "DIABETES": -0.6,             # away from GAS
    "AGE": -0.4,                  # older (>50) away from GAS
    "creatinine_preop": 1.0,      # strongest pointer toward GAS (above ~110)
    "hgb_preop": -0.2,            # weak; assumed lower Hb slightly toward GAS
}
SURGERY_COEF = -2.1
TARGET_AUC = 0.83


def _exact_binary(n, prob, rng, score=None):
    """Exactly round(prob*n) ones; if `score` given, ones go to the top-k of score + logistic noise."""
    k = int(round(prob * n))
    z = rng.logistic(size=n) + (0 if score is None else score)
    out = np.zeros(n, dtype=int)
    out[np.argsort(-z)[:k]] = 1
    return out


def draw_features(n: int, rng: np.random.Generator, exact: bool = False) -> pd.DataFrame:
    """exact=True: binary/categorical counts match the published percentages exactly (sample)."""
    def z(size):  # standard normal draws; standardised in-sample when exact=True
        x = rng.normal(0, 1, size)
        return (x - x.mean()) / x.std() if exact else x

    age = np.clip(AGE_MEAN + AGE_SD * z(n), 18, 100).round()
    # Diabetes: mild positive dependence on age. Intercept solved so the rate is ~24%.
    age_z = (age - AGE_MEAN) / AGE_SD
    b_age = 0.5
    a = brentq(lambda a_: expit(a_ + b_age * age_z).mean() - P_DIABETES, -5, 5)
    probs = [P_UPPER, P_LOWER, P_ANOG, 1 - P_UPPER - P_LOWER - P_ANOG]
    if exact:
        diabetes = _exact_binary(n, P_DIABETES, rng, score=b_age * age_z)
        counts = [int(round(q * n)) for q in probs[:3]]
        site = rng.permutation(np.repeat([0, 1, 2, 3], counts + [n - sum(counts)]))
        surgery = _exact_binary(n, P_SURGERY, rng)
    else:
        diabetes = rng.binomial(1, expit(a + b_age * age_z))
        site = rng.choice(4, size=n, p=probs)  # arrival site: mutually exclusive
        surgery = rng.binomial(1, P_SURGERY, n)
    creat = np.clip(np.exp(CREAT_MU + CREAT_SIGMA * z(n)), 30, 1500).round(1)
    hgb = np.clip(HGB_MEAN + HGB_SD * z(n), 4.0, 20.0).round(1)
    return pd.DataFrame({
        "AGE": age.astype(int),
        "DIABETES": diabetes,
        "UPPER_ARM_ARRIVAL": (site == 0).astype(int),
        "LOWER_ARM_ARRIVAL": (site == 1).astype(int),
        "ANOGENITAL_ARRIVAL": (site == 2).astype(int),
        "SURGERY_BEFORE": surgery,
        "creatinine_preop": creat,
        "hgb_preop": hgb,
    })


def design(df: pd.DataFrame, variant: str) -> dict[str, np.ndarray]:
    """Transformed feature terms that enter the hidden linear predictor."""
    if variant == "threshold":
        creat_term = (df["creatinine_preop"].to_numpy() > 110).astype(float) * 1.5
    else:
        creat_term = np.log(df["creatinine_preop"].to_numpy() / 110) / CREAT_SIGMA
    return {
        "UPPER_ARM_ARRIVAL": df["UPPER_ARM_ARRIVAL"].to_numpy(float),
        "LOWER_ARM_ARRIVAL": df["LOWER_ARM_ARRIVAL"].to_numpy(float),
        "ANOGENITAL_ARRIVAL": df["ANOGENITAL_ARRIVAL"].to_numpy(float),
        "DIABETES": df["DIABETES"].to_numpy(float),
        "AGE": (df["AGE"].to_numpy(float) - 50) / 15,
        "creatinine_preop": creat_term,
        "hgb_preop": (df["hgb_preop"].to_numpy(float) - HGB_MEAN) / HGB_SD,
    }


def linear_predictor(terms, df, strength, intercept):
    eta = intercept + SURGERY_COEF * df["SURGERY_BEFORE"].to_numpy(float)
    for k, w in BASE_WEIGHTS.items():
        eta = eta + strength * w * terms[k]
    return eta


def tune(variant: str, seed: int, n_pop: int = 200_000):
    """Tune (strength, intercept) on a large population so rate=30.8% and true AUC~0.83."""
    rng = np.random.default_rng(seed + 1)
    pop = draw_features(n_pop, rng)
    terms = design(pop, variant)
    u = rng.random(n_pop)  # common random numbers -> smooth objective

    def intercept_for(s):
        return brentq(lambda b: expit(linear_predictor(terms, pop, s, b)).mean() - GAS_RATE, -10, 10)

    def auc_for(s):
        p = expit(linear_predictor(terms, pop, s, intercept_for(s)))
        return roc_auc_score((u < p).astype(int), p)

    strength = brentq(lambda s: auc_for(s) - TARGET_AUC, 0.05, 5.0, xtol=1e-3)
    intercept = intercept_for(strength)
    return strength, intercept, auc_for(strength)


def generate(n: int = config.N_PATIENTS, seed: int = config.SEED, variant: str = "logistic"):
    strength, intercept, pop_auc = tune(variant, seed)
    rng = np.random.default_rng(seed)
    df = draw_features(n, rng, exact=True)
    # Latent-variable form of the logistic model: GAS=1 for the top-k of eta + Logistic noise,
    # k = round(30.8% * n). Equivalent to re-solving the intercept on this sample.
    eta0 = linear_predictor(design(df, variant), df, strength, 0.0)
    latent = eta0 + rng.logistic(size=n)
    k = int(round(GAS_RATE * n))
    cut = np.sort(latent)[::-1][k - 1:k + 1].mean()
    df[config.TARGET] = (latent > cut).astype(int)
    sample_intercept = -cut
    p_true = expit(eta0 + sample_intercept)
    sample_auc = roc_auc_score(df[config.TARGET], p_true)
    # Pure-noise decoys (independent of everything).
    df["NOISE_1"] = rng.normal(0, 1, n).round(3)
    df["NOISE_2"] = rng.uniform(0, 100, n).round(2)
    df["NOISE_3"] = rng.binomial(1, 0.3, n)
    # MCAR missingness in the two lab values.
    for col, frac in MISSING.items():
        idx = rng.choice(n, size=int(round(frac * n)), replace=False)
        df.loc[idx, col] = np.nan
    truth = {
        "SYNTHETIC": True,
        "seed": seed,
        "n": n,
        "variant": variant,
        "intercept": intercept,
        "intercept_sample_realised": float(sample_intercept),
        "strength_multiplier": strength,
        "effective_coefficients": {
            **{k: strength * w for k, w in BASE_WEIGHTS.items()},
            "SURGERY_BEFORE": SURGERY_COEF,
        },
        "term_definitions": {
            "AGE": "(AGE-50)/15",
            "creatinine_preop": ("1.5 * 1[creat>110]" if variant == "threshold"
                                 else "log(creat/110)/0.67"),
            "hgb_preop": "(hgb-11.4)/3.04",
            "binaries": "0/1",
        },
        "decoys": config.DECOYS,
        "missing_fraction": MISSING,
        "best_achievable_auc_population": pop_auc,
        "best_achievable_auc_sample": sample_auc,
        "gas_rate_target": GAS_RATE,
        "gas_rate_sample": float(df[config.TARGET].mean()),
        "note": "Magnitudes are assumptions; directions from paper SHAP plots. Not real data.",
    }
    return df, truth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["logistic", "threshold"], default="logistic")
    ap.add_argument("--n", type=int, default=config.N_PATIENTS)
    ap.add_argument("--seed", type=int, default=config.SEED)
    args = ap.parse_args()
    df, truth = generate(args.n, args.seed, args.variant)
    suffix = "" if args.variant == "logistic" else f"_{args.variant}"
    csv = config.DATA_DIR / f"nsti_gas_SYNTHETIC{suffix}.csv"
    gt = config.DATA_DIR / f"ground_truth{suffix}.json"
    df.to_csv(csv, index=False)
    gt.write_text(json.dumps(truth, indent=2))
    print(f"wrote {csv.name} ({len(df)} rows), {gt.name}")
    print(json.dumps({k: truth[k] for k in ["strength_multiplier", "intercept",
                                            "best_achievable_auc_population",
                                            "best_achievable_auc_sample", "gas_rate_sample"]},
                     indent=2))


if __name__ == "__main__":
    main()
