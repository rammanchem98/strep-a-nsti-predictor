import json

import pandas as pd

from src import config
from src.data_utils import load_data, load_splits, split_data
from src.features import make_preprocessor


def test_saved_splits_match_split_function():
    saved = load_splits()
    fresh = split_data(load_data())
    for a, b in zip(saved, fresh):
        pd.testing.assert_frame_equal(a, b, check_dtype=False)


def test_preprocessor_imputes_and_scales():
    train, val, _ = load_splits()
    pre = make_preprocessor().fit(train[config.FEATURES])
    out = pre.transform(val[config.FEATURES])
    assert out.notna().all().all()
    assert out.shape == (len(val), 8)
    assert set(out.columns) == set(config.FEATURES)


def test_boruta_rejects_decoys():
    res = pd.read_csv(config.ARTIFACT_DIR / "boruta_SYNTHETIC.csv", index_col=0)
    assert not res.loc[config.DECOYS, "selected"].any()
    sel = json.loads((config.ARTIFACT_DIR / "boruta_selected.json").read_text())["selected"]
    assert set(sel) <= set(config.FEATURES)
