"""Cleaning, deterministic splits and baselines for the configured team dataset."""

from pathlib import Path

import numpy as np
import pandas as pd

from app.sensors import BOUNDS, FEATURES

DATA = Path(__file__).resolve().parent / "data/team_02_train.csv"


def clean(raw):
    frame = raw[FEATURES + ["days_left"]].apply(pd.to_numeric, errors="coerce")
    invalid = ~np.isfinite(frame)
    for col, (lo, hi) in BOUNDS.items():
        invalid[col] |= ~frame[col].between(lo, hi)
    invalid.wifi_rssi_dbm |= frame.wifi_rssi_dbm.eq(0)
    invalid.days_left |= frame.days_left.lt(0)
    return frame.loc[~invalid.any(axis=1)], invalid


def split(frame, seed=0):
    idx = np.random.default_rng(seed).permutation(len(frame))
    n = len(idx) // 5
    return frame.iloc[idx[n:]], frame.iloc[idx[:n]]


def baselines(train, val):
    def design(frame):
        return np.column_stack((frame[FEATURES], np.ones(len(frame))))

    coef = np.linalg.lstsq(design(train), train.days_left, rcond=None)[0]
    return train.days_left.mean(), design(val) @ coef
