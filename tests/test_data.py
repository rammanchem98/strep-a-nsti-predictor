import json
import math

import numpy as np
import pytest

from data.generate_synthetic import generate
from src import config
from src.data_utils import load_data, split_data


@pytest.fixture(scope="module")
def df():
    return load_data()


def se_prop(p, n):
    return math.sqrt(p * (1 - p) / n)


def test_file_is_reproducible(df):
    regen, _ = generate(seed=config.SEED)
    assert np.allclose(regen[config.FEATURES + [config.TARGET]].fillna(-1).to_numpy(),
                       df[config.FEATURES + [config.TARGET]].fillna(-1).to_numpy())


def test_shape_and_columns(df):
    assert len(df) == 409
    assert set(config.FEATURES + config.DECOYS + [config.TARGET]) <= set(df.columns)


def test_gas_rate(df):
    assert df[config.TARGET].sum() == 126


@pytest.mark.parametrize("col,p", [("UPPER_ARM_ARRIVAL", 0.117), ("LOWER_ARM_ARRIVAL", 0.137),
                                   ("ANOGENITAL_ARRIVAL", 0.350), ("SURGERY_BEFORE", 0.188),
                                   ("DIABETES", 0.240)])
def test_binary_marginals(df, col, p):
    assert abs(df[col].mean() - p) < 2 * se_prop(p, len(df))


def test_sites_mutually_exclusive(df):
    s = df[["UPPER_ARM_ARRIVAL", "LOWER_ARM_ARRIVAL", "ANOGENITAL_ARRIVAL"]].sum(axis=1)
    assert s.max() <= 1


@pytest.mark.parametrize("col,mean,sd", [("AGE", 58.7, 15.1), ("creatinine_preop", 161.2, 121.3),
                                         ("hgb_preop", 11.4, 3.04)])
def test_continuous_marginals(df, col, mean, sd):
    x = df[col].dropna()
    assert abs(x.mean() - mean) < 2 * sd / math.sqrt(len(x))
    assert 0.8 * sd < x.std() < 1.2 * sd


def test_ranges(df):
    assert df.AGE.between(18, 100).all()
    assert df.hgb_preop.dropna().between(4, 20).all()
    assert abs(df.creatinine_preop.median() - 129) < 15


def test_missingness(df):
    for col in ["creatinine_preop", "hgb_preop"]:
        assert 0.05 <= df[col].isna().mean() <= 0.15
    assert df[[c for c in config.FEATURES if c not in ("creatinine_preop", "hgb_preop")]].notna().all().all()


def test_signal_directions(df):
    g = df.groupby(config.TARGET).mean(numeric_only=True)
    assert g.loc[1, "UPPER_ARM_ARRIVAL"] > g.loc[0, "UPPER_ARM_ARRIVAL"]
    assert g.loc[1, "ANOGENITAL_ARRIVAL"] < g.loc[0, "ANOGENITAL_ARRIVAL"]
    assert g.loc[1, "SURGERY_BEFORE"] < g.loc[0, "SURGERY_BEFORE"]
    assert g.loc[1, "creatinine_preop"] > g.loc[0, "creatinine_preop"]


def test_ground_truth():
    gt = json.loads(config.GROUND_TRUTH.read_text())
    assert gt["SYNTHETIC"] is True and gt["seed"] == config.SEED
    assert abs(gt["best_achievable_auc_population"] - 0.83) < 0.01
    assert gt["effective_coefficients"]["SURGERY_BEFORE"] == -2.1


def test_split_stratification(df):
    train, val, test = split_data(df)
    assert (len(train), len(val), len(test)) == (245, 82, 82)
    for part in (train, val, test):
        assert abs(part[config.TARGET].mean() - 126 / 409) < 0.01
    ids = [set(map(tuple, p[config.FEATURES].fillna(-1).to_numpy())) for p in (train, val, test)]
    assert not (ids[0] & ids[2]) and not (ids[1] & ids[2])
