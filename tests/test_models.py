import pickle

import joblib
import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src import config
from src.data_utils import load_splits, xy
from src.features import make_preprocessor
from src.models import TorchMLPClassifier


@pytest.fixture(scope="module")
def data():
    train, val, _ = load_splits()
    return xy(train), xy(val)


@pytest.mark.parametrize("clf", [LogisticRegression(max_iter=1000),
                                 RandomForestClassifier(n_estimators=20, random_state=0)])
def test_sklearn_pipelines_fit_predict(data, clf):
    (X, y), (Xv, _) = data
    pipe = Pipeline([("pre", make_preprocessor()), ("clf", clf)]).fit(X, y)
    p = pipe.predict_proba(Xv)
    assert p.shape == (len(Xv), 2)
    assert np.allclose(p.sum(1), 1)
    assert set(pipe.predict(Xv)) <= {0, 1}


def test_mlp_fit_predict_and_pickle(data):
    (X, y), (Xv, yv) = data
    pre = make_preprocessor().fit(X)
    Xt, Xvt = pre.transform(X).to_numpy(), pre.transform(Xv).to_numpy()
    clf = TorchMLPClassifier(max_epochs=50, patience=10).fit(Xt, y, Xvt, yv)
    p = clf.predict_proba(Xvt)
    assert p.shape == (len(Xv), 2) and ((p >= 0) & (p <= 1)).all()
    assert abs(clf.pos_weight_ - (y == 0).sum() / (y == 1).sum()) < 1e-9
    clone = pickle.loads(pickle.dumps(clf))
    assert np.allclose(clone.predict_proba(Xvt), p)


def test_mlp_is_deterministic(data):
    (X, y), (Xv, yv) = data
    pre = make_preprocessor().fit(X)
    Xt, Xvt = pre.transform(X).to_numpy(), pre.transform(Xv).to_numpy()
    a = TorchMLPClassifier(max_epochs=30).fit(Xt, y).predict_proba(Xvt)
    b = TorchMLPClassifier(max_epochs=30).fit(Xt, y).predict_proba(Xvt)
    assert np.allclose(a, b)


def test_served_pipeline_handles_missing_labs(data):
    _, (Xv, _) = data
    pipe = joblib.load(config.SERVED_MODEL)
    X = Xv.head(5).copy()
    X.loc[:, ["creatinine_preop", "hgb_preop"]] = np.nan
    p = pipe.predict_proba(X)[:, 1]
    assert p.shape == (5,) and np.isfinite(p).all()
