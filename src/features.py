"""Preprocessing (authors' recipe) and Boruta feature selection on the SYNTHETIC train split."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from src import config


def make_preprocessor(numeric=None, binary=None, seed: int = config.SEED) -> ColumnTransformer:
    """numeric -> MinMaxScaler + IterativeImputer(sample_posterior); binary -> KNNImputer(k=5)."""
    numeric = list(config.NUMERIC if numeric is None else numeric)
    binary = list(config.BINARY if binary is None else binary)
    num = Pipeline([
        ("scale", MinMaxScaler()),
        ("impute", IterativeImputer(sample_posterior=True, max_iter=10, n_nearest_features=5,
                                    random_state=seed)),
    ])
    ct = ColumnTransformer([("num", num, numeric), ("bin", KNNImputer(n_neighbors=5), binary)],
                           verbose_feature_names_out=False)
    return ct.set_output(transform="pandas")


def run_boruta(train: pd.DataFrame, n_iter: int = 30, frac: float = 0.8, seed: int = config.SEED):
    """Boruta on 80% subsamples of train, `n_iter` times; keep features confirmed in >50% of runs."""
    from boruta import BorutaPy

    cols = config.FEATURES + config.DECOYS
    numeric = config.NUMERIC + ["NOISE_1", "NOISE_2"]
    binary = config.BINARY + ["NOISE_3"]
    rng = np.random.default_rng(seed)
    confirmed = pd.Series(0, index=cols, dtype=int)
    tentative = pd.Series(0, index=cols, dtype=int)
    for i in range(n_iter):
        idx = rng.choice(len(train), size=int(frac * len(train)), replace=False)
        sub = train.iloc[idx]
        pre = make_preprocessor(numeric, binary, seed=seed + i)
        X = pre.fit_transform(sub[cols])[cols]
        y = sub[config.TARGET].to_numpy()
        rf = RandomForestClassifier(n_jobs=-1, class_weight="balanced", max_depth=5,
                                    random_state=seed + i)
        bor = BorutaPy(rf, n_estimators="auto", max_iter=100, random_state=seed + i, verbose=0)
        bor.fit(X.to_numpy(), y)
        confirmed += bor.support_.astype(int)
        tentative += bor.support_weak_.astype(int)
    freq = (confirmed / n_iter).rename("confirmed_frac")
    out = pd.concat([freq, (tentative / n_iter).rename("tentative_frac")], axis=1)
    out["selected"] = out["confirmed_frac"] > 0.5
    return out.sort_values("confirmed_frac", ascending=False)


def main():
    from src.data_utils import load_data, save_splits, split_data

    df = load_data()
    train, val, test = split_data(df)
    save_splits(train, val, test)
    res = run_boruta(train)
    print(res.to_string())
    config.ARTIFACT_DIR.mkdir(exist_ok=True)
    res.to_csv(config.ARTIFACT_DIR / "boruta_SYNTHETIC.csv")
    selected = res.index[res["selected"]].tolist()
    (config.ARTIFACT_DIR / "boruta_selected.json").write_text(json.dumps(
        {"selected": selected, "n_iter": 30, "subsample": 0.8, "rule": "confirmed in >50%"}, indent=2))


if __name__ == "__main__":
    main()
