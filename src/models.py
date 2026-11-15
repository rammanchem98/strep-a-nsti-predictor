"""Model definitions. Every served model is ONE sklearn Pipeline: preprocessing + classifier."""
from __future__ import annotations

import copy
import io

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from torch import nn

from src import config
from src.features import make_preprocessor

RF_GRID = {"clf__n_estimators": [100, 300, 700], "clf__max_depth": [2, 4, 6],
           "clf__max_features": [2, 4, 6]}
LOGREG_GRID = {"clf__penalty": [None, "l2"], "clf__C": [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
MLP_GRID = {"hidden": [(16, 8), (32, 16)], "lr": [1e-3, 3e-3], "dropout": [0.1, 0.3],
            "weight_decay": [1e-4, 1e-3]}


def logreg_search(seed=config.SEED) -> GridSearchCV:
    pipe = Pipeline([("pre", make_preprocessor(seed=seed)),
                     ("clf", LogisticRegression(class_weight="balanced", max_iter=5000))])
    return GridSearchCV(pipe, LOGREG_GRID, scoring="balanced_accuracy", n_jobs=-1,
                        cv=StratifiedKFold(5, shuffle=True, random_state=seed))


def rf_search(seed=config.SEED) -> GridSearchCV:
    pipe = Pipeline([("pre", make_preprocessor(seed=seed)),
                     ("clf", RandomForestClassifier(class_weight="balanced", random_state=seed))])
    return GridSearchCV(pipe, RF_GRID, scoring="balanced_accuracy", n_jobs=-1,
                        cv=StratifiedKFold(5, shuffle=True, random_state=seed))


class _Net(nn.Module):
    def __init__(self, n_in, hidden, dropout):
        super().__init__()
        layers, prev = [], n_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class TorchMLPClassifier(ClassifierMixin, BaseEstimator):
    """Small PyTorch MLP with an sklearn interface (so it can sit inside a Pipeline).

    fit(X, y, X_val=None, y_val=None): BCEWithLogitsLoss(pos_weight=n_neg/n_pos), AdamW weight decay,
    full-batch training, early stopping on validation loss (restores best weights).
    """

    def __init__(self, hidden=(16, 8), lr=1e-3, dropout=0.1, weight_decay=1e-4,
                 max_epochs=2000, patience=100, seed=config.SEED):
        self.hidden = hidden
        self.lr = lr
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.patience = patience
        self.seed = seed

    def fit(self, X, y, X_val=None, y_val=None):
        torch.manual_seed(self.seed)
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = X.shape[1]
        self.pos_weight_ = float((y == 0).sum() / max((y == 1).sum(), 1))
        net = _Net(X.shape[1], tuple(self.hidden), self.dropout)
        opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(self.pos_weight_))
        Xt, yt = torch.from_numpy(X), torch.from_numpy(y)
        has_val = X_val is not None
        if has_val:
            Xv = torch.from_numpy(np.asarray(X_val, dtype=np.float32))
            yv = torch.from_numpy(np.asarray(y_val, dtype=np.float32))
        best, best_state, wait, self.best_epoch_ = np.inf, None, 0, 0
        for epoch in range(self.max_epochs):
            net.train()
            opt.zero_grad()
            loss = loss_fn(net(Xt), yt)
            loss.backward()
            opt.step()
            if has_val:
                net.eval()
                with torch.no_grad():
                    vl = loss_fn(net(Xv), yv).item()
                if vl < best - 1e-5:
                    best, best_state, wait, self.best_epoch_ = vl, copy.deepcopy(net.state_dict()), 0, epoch
                else:
                    wait += 1
                    if wait >= self.patience:
                        break
        if best_state is not None:
            net.load_state_dict(best_state)
        self.best_val_loss_ = best
        self.net_ = net.eval()
        return self

    def predict_proba(self, X):
        X = torch.from_numpy(np.asarray(X, dtype=np.float32))
        with torch.no_grad():
            p = torch.sigmoid(self.net_(X)).numpy()
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    # Pickle the network as a state_dict blob (portable across processes).
    def __getstate__(self):
        state = self.__dict__.copy()
        if "net_" in state:
            buf = io.BytesIO()
            torch.save(state.pop("net_").state_dict(), buf)
            state["_net_bytes"] = buf.getvalue()
        return state

    def __setstate__(self, state):
        blob = state.pop("_net_bytes", None)
        self.__dict__.update(state)
        if blob is not None:
            net = _Net(self.n_features_in_, tuple(self.hidden), self.dropout)
            net.load_state_dict(torch.load(io.BytesIO(blob), weights_only=True))
            self.net_ = net.eval()


def mlp_search(X_train, y_train, X_val, y_val, seed=config.SEED):
    """Grid over width/lr/dropout/weight decay; early stopping and selection on val balanced accuracy.

    Returns (fitted Pipeline, best params, list of (params, val_bal_acc)).
    """
    from itertools import product

    pre = make_preprocessor(seed=seed).fit(X_train)
    Xt, Xv = pre.transform(X_train).to_numpy(), pre.transform(X_val).to_numpy()
    results, best = [], (-1, None, None)
    keys = list(MLP_GRID)
    for combo in product(*MLP_GRID.values()):
        params = dict(zip(keys, combo))
        clf = TorchMLPClassifier(seed=seed, **params).fit(Xt, y_train, Xv, y_val)
        score = balanced_accuracy_score(y_val, clf.predict(Xv))
        results.append((params, score, clf.best_epoch_))
        if score > best[0]:
            best = (score, params, clf)
    pipe = Pipeline([("pre", pre), ("clf", best[2])])
    return pipe, best[1], results
