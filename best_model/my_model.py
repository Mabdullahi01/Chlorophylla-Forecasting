"""Grading interface: one-day-ahead chlorophyll-a prediction with a linear ARX model."""

from __future__ import annotations

from pathlib import Path
import pickle
from typing import Any

import numpy as np
import pandas as pd

from utils import (
    DATE, TARGET, build_features, inverse_target, linear_predict, prepare_raw,
    standardize,
)


def load_model(model_path: str | Path) -> Any:
    """Load the dictionary saved by training_code/train.py."""
    with Path(model_path).open("rb") as handle:
        return pickle.load(handle)


def predict(model: Any, history_df: pd.DataFrame, day_features_df: pd.DataFrame) -> np.ndarray:
    """Predict today's chlorophyll-a (mg/m3) as a NumPy array of shape (1,).

    history_df: every observation strictly before today, including chlorophyll.
    day_features_df: today's date and 12 exogenous variables, no chlorophyll.
    Neither input is modified: prepare_raw works on a copy.
    """
    config = model["config"]


    today = day_features_df.copy()
    today[TARGET] = np.nan
    combined = pd.concat([history_df, today], ignore_index=True, sort=False)
    frame = prepare_raw(combined)


    today_date = pd.to_datetime(day_features_df[DATE].iloc[0])
    features = build_features(frame, config)
    row = features[frame[DATE] == today_date].iloc[[-1]]
    X = row[model["feature_names"]].to_numpy(dtype=float)


    Z = standardize(X, model["mu"], model["sigma"])


    y_model = linear_predict(Z, model["coef"], model["intercept"])
    y = inverse_target(y_model, config["target_transform"], model["smearing"])


    value = float(y[0])
    if not np.isfinite(value):
        value = model["fallback_value"]
    return np.asarray([max(value, 0.0)], dtype=float)
