"""Loading and splitting the SYNTHETIC cohort."""
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from src import config


def load_data(path=config.SYNTH_CSV) -> pd.DataFrame:
    df = pd.read_csv(path)
    assert config.TARGET in df.columns
    return df


def split_data(df: pd.DataFrame, seed: int = config.SEED):
    """Stratified 60/20/20 train/val/test."""
    train, temp = train_test_split(df, test_size=0.4, stratify=df[config.TARGET], random_state=seed)
    val, test = train_test_split(temp, test_size=0.5, stratify=temp[config.TARGET], random_state=seed)
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def save_splits(train, val, test, out=config.SPLIT_DIR):
    out.mkdir(parents=True, exist_ok=True)
    for name, d in [("train", train), ("val", val), ("test", test)]:
        d.to_csv(out / f"{name}_SYNTHETIC.csv", index=False)


def load_splits(out=config.SPLIT_DIR):
    return tuple(pd.read_csv(out / f"{n}_SYNTHETIC.csv") for n in ("train", "val", "test"))


def xy(df: pd.DataFrame, features=None):
    features = features or config.FEATURES
    return df[features], df[config.TARGET].to_numpy()
