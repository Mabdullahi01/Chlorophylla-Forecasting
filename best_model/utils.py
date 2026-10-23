"""Shared explicit feature construction and numerical transformations."""

from __future__ import annotations

import numpy as np
import pandas as pd

DATE = "date"
TARGET = "chlorophyll_a_mg_m3"
EXOGENOUS = [
    "sst_c", "par_umol_m2_s", "nitrate_umol_l", "wind_speed_m_s",
    "upwelling_index", "mixed_layer_depth_m", "salinity_psu",
    "current_speed_m_s", "river_discharge_index", "cloud_fraction",
    "surface_pressure_hpa", "turbidity_ntu",
]
LOG_FLOOR = 0.05  # mg/m3; keeps log() finite if a value is ever ~0


def despike_causal(values: pd.Series, window: int, k: float) -> pd.Series:
    """Replace isolated spikes using only the current and PAST values.

    A value is a spike when it is more than k robust standard deviations
    (1.4826 * MAD) away from the median of the last `window` days. It is then
    replaced by that median. Because only past data is used, applying this to
    the full training series gives exactly the same values as applying it to
    the 2024 history at prediction time: no look-ahead.
    """
    median = values.rolling(window, min_periods=1).median()
    deviation = (values - median).abs()
    mad = deviation.rolling(window, min_periods=1).median()
    spikes = deviation > k * 1.4826 * mad
    return values.where(~spikes, median)


def despike_centered(values: pd.Series, window: int, k: float) -> pd.Series:
    """Same rule with a centred window (past AND future days).

    Used only to clean TRAINING LABELS, where the whole series is known.
    Never used to build prediction inputs.
    """
    median = values.rolling(window, center=True, min_periods=1).median()
    deviation = (values - median).abs()
    mad = deviation.rolling(window, center=True, min_periods=1).median()
    spikes = deviation > k * 1.4826 * mad
    return values.where(~spikes, median)


def forward_target(values, transform: str):
    """mg/m3 -> modelling scale."""
    if transform == "log":
        return np.log(np.maximum(values, LOG_FLOOR))
    return values


def inverse_target(values, transform: str, smearing: float = 1.0):
    """Modelling scale -> mg/m3.

    exp(mean of log) estimates the median, not the mean, so the log
    back-transform is multiplied by Duan's smearing factor
    mean(exp(training residuals)), computed in train.py.
    """
    if transform == "log":
        return np.exp(values) * smearing
    return values


def prepare_raw(raw: pd.DataFrame) -> pd.DataFrame:
    """Copy, parse dates, sort chronologically, fill gaps in exogenous inputs.

    Missing exogenous values (none in the released data, but possible in
    the test set) are carried forward from the last observation, which
    uses only past information.
    """
    frame = raw.copy()
    frame[DATE] = pd.to_datetime(frame[DATE])
    frame = frame.sort_values(DATE).reset_index(drop=True)
    for col in EXOGENOUS:
        frame[col] = pd.to_numeric(frame[col], errors="coerce").ffill()
    if TARGET in frame:
        frame[TARGET] = pd.to_numeric(frame[TARGET], errors="coerce")
    else:
        frame[TARGET] = np.nan
    return frame


def build_features(frame: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Return one feature row per date of `frame` (already prepared).

    Row t uses chlorophyll from t-1 ... t-p (never t) and exogenous values
    from t, t-1, ... t-q. Columns are ordered as config["lags"] lists them,
    followed by the two seasonal features.
    """
    chl = frame[TARGET].ffill()  # a missing past day reuses the previous one
    if config["spike_k"] is not None:
        chl = despike_causal(chl, config["spike_window"], config["spike_k"])
    chl = forward_target(chl, config["target_transform"])

    columns: dict[str, pd.Series | np.ndarray] = {}
    for col, lags in config["lags"].items():
        source = chl if col == TARGET else frame[col]
        for lag in lags:
            if col == TARGET and lag == 0:
                raise ValueError("target lag 0 is forbidden: it leaks the answer")
            columns[f"{col}__lag_{lag}"] = source.shift(lag)
    day = frame[DATE].dt.dayofyear.to_numpy()
    columns["annual_sin"] = np.sin(2 * np.pi * day / 365.25)
    columns["annual_cos"] = np.cos(2 * np.pi * day / 365.25)
    return pd.DataFrame(columns, index=frame.index)


def max_lag(config: dict) -> int:
    return max(max(lags) for lags in config["lags"].values())


def fit_standardizer(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Column means and standard deviations, computed on TRAINING rows only."""
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma[sigma == 0] = 1.0  # a constant column would otherwise divide by 0
    return mu, sigma


def standardize(X: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """(X - mu) / sigma; any remaining NaN becomes 0, i.e. the training mean."""
    Z = (X - mu) / sigma
    return np.where(np.isfinite(Z), Z, 0.0)


def linear_predict(Z: np.ndarray, coef: np.ndarray, intercept: float) -> np.ndarray:
    """y = c + Z @ theta: the ARX regression in explicit form (Eq. 2 of the manual)."""
    return intercept + Z @ coef
