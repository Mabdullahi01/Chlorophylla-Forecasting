"""Train and select a linear ARX model using chronological validation."""

from __future__ import annotations

import argparse
import itertools
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LinearRegression, Ridge

# The feature code lives in best_model/utils.py so that training and
# prediction share one implementation.
BEST_MODEL_DIR = Path(__file__).resolve().parent.parent / "best_model"
sys.path.insert(0, str(BEST_MODEL_DIR))

from utils import (  # noqa: E402
    DATE, TARGET, build_features, despike_centered, fit_standardizer,
    forward_target, inverse_target, linear_predict, max_lag, prepare_raw,
    standardize,
)

VALIDATION_YEARS = [2020, 2021, 2022]
TEST_YEAR = 2023
SPIKE_WINDOW = 31


# "provided": DEFAULT_LAGS from build_lagged_features.py.
# "explored": from the ACF/PACF and cross-correlation analysis in exploration/.
STORM = ["upwelling_index", "nitrate_umol_l", "wind_speed_m_s",
         "current_speed_m_s", "mixed_layer_depth_m"]
WEAK = ["sst_c", "par_umol_m2_s", "salinity_psu", "river_discharge_index",
        "cloud_fraction", "turbidity_ntu"]


def lag_set(p: int, storm_q: int, weak_q: int | None, pressure_q: int | None) -> dict:
    lags = {TARGET: list(range(1, p + 1))}
    for col in STORM:
        lags[col] = list(range(0, storm_q + 1))
    if weak_q is not None:
        for col in WEAK:
            lags[col] = list(range(0, weak_q + 1))
    if pressure_q is not None:
        lags["surface_pressure_hpa"] = list(range(0, pressure_q + 1))
    return lags


LAG_CANDIDATES = {
    "provided": {
        TARGET: list(range(1, 15)),
        "sst_c": list(range(0, 15)), "par_umol_m2_s": list(range(0, 8)),
        "nitrate_umol_l": list(range(0, 15)), "wind_speed_m_s": list(range(0, 8)),
        "upwelling_index": list(range(0, 15)), "mixed_layer_depth_m": list(range(0, 8)),
        "salinity_psu": list(range(0, 8)), "current_speed_m_s": list(range(0, 8)),
        "river_discharge_index": list(range(0, 8)), "cloud_fraction": list(range(0, 8)),
        "surface_pressure_hpa": list(range(0, 8)), "turbidity_ntu": list(range(0, 8)),
    },
    "explored":         lag_set(p=7, storm_q=5, weak_q=3, pressure_q=None),
    "explored_short":   lag_set(p=3, storm_q=4, weak_q=1, pressure_q=None),
    "explored_no_weak": lag_set(p=7, storm_q=5, weak_q=None, pressure_q=None),
    "explored_long":    lag_set(p=14, storm_q=7, weak_q=7, pressure_q=7),
}

TARGET_TRANSFORMS = ["none", "log"]
SPIKE_KS = [None, 8.0]  # None = no despiking


def model_candidates(transform: str) -> dict:
    """Estimators to compare. LASSO's penalty scale depends on the target scale."""
    models = {"ols": lambda: LinearRegression()}
    for alpha in [0.1, 1, 10, 100, 1000]:
        models[f"ridge_{alpha:g}"] = lambda a=alpha: Ridge(alpha=a)
    lasso_alphas = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2] if transform == "log" else [3e-4, 1e-3, 3e-3, 1e-2, 3e-2]
    for alpha in lasso_alphas:
        models[f"lasso_{alpha:g}"] = lambda a=alpha: Lasso(alpha=a, max_iter=50000)
    return models


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """R2, RMSE and MAE in mg/m3, written out explicitly."""
    residual = y_true - y_pred
    ss_res = float(np.sum(residual ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return {"r2": 1.0 - ss_res / ss_tot,
            "rmse": float(np.sqrt(np.mean(residual ** 2))),
            "mae": float(np.mean(np.abs(residual)))}


def fit(frame: pd.DataFrame, features: pd.DataFrame, train_mask: np.ndarray,
        config: dict, make_model) -> dict:
    """Fit one configuration on the rows in train_mask; return plain parameters."""
    X_train = features[train_mask].to_numpy(dtype=float)
    mu, sigma = fit_standardizer(X_train)
    Z_train = standardize(X_train, mu, sigma)

    # Training labels: despiked with a centred window computed on the
    # training years only (labels never feed back into inputs).
    labels = frame[TARGET]
    if config["spike_k"] is not None:
        last_train_date = frame.loc[train_mask, DATE].max()
        known = (frame[DATE] <= last_train_date).to_numpy()  # no validation days
        labels = despike_centered(labels[known], SPIKE_WINDOW, config["spike_k"])
        labels = labels.reindex(frame.index)
    y_train = forward_target(labels[train_mask].to_numpy(dtype=float), config["target_transform"])

    estimator = make_model()
    estimator.fit(Z_train, y_train)
    coef = np.asarray(estimator.coef_, dtype=float)
    intercept = float(estimator.intercept_)

    smearing = 1.0
    if config["target_transform"] == "log":
        residuals = y_train - linear_predict(Z_train, coef, intercept)
        smearing = float(np.mean(np.exp(residuals)))
    return {"coef": coef, "intercept": intercept, "mu": mu, "sigma": sigma,
            "smearing": smearing}


def predict_rows(features: pd.DataFrame, mask: np.ndarray, params: dict, config: dict) -> np.ndarray:
    Z = standardize(features[mask].to_numpy(dtype=float), params["mu"], params["sigma"])
    y = inverse_target(linear_predict(Z, params["coef"], params["intercept"]),
                       config["target_transform"], params["smearing"])
    return np.maximum(y, 0.0)


def masks(frame: pd.DataFrame, config: dict, eval_year: int) -> tuple[np.ndarray, np.ndarray]:
    """Train = all usable rows before eval_year; eval = rows of eval_year."""
    year = frame[DATE].dt.year.to_numpy()
    usable = np.arange(len(frame)) >= max_lag(config)  # first rows lack full lags
    return usable & (year < eval_year), year == eval_year


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()


    if args.data.suffix.lower() == ".xlsx":
        raw = pd.read_excel(args.data, sheet_name="daily_data")
    else:
        raw = pd.read_csv(args.data)
    frame = prepare_raw(raw)
    y_true = frame[TARGET].to_numpy(dtype=float)
    print(f"loaded {len(frame)} days: {frame[DATE].min().date()} to {frame[DATE].max().date()}")

    # Baseline: persistence, chlorophyll(t) = chlorophyll(t-1).
    persistence = frame[TARGET].shift(1).to_numpy(dtype=float)
    for yr in VALIDATION_YEARS + [TEST_YEAR]:
        m = frame[DATE].dt.year.to_numpy() == yr
        s = metrics(y_true[m], persistence[m])
        print(f"persistence baseline {yr}: R2={s['r2']:.3f} RMSE={s['rmse']:.3f} MAE={s['mae']:.3f}")


    results = []
    for lag_name, transform, spike_k in itertools.product(LAG_CANDIDATES, TARGET_TRANSFORMS, SPIKE_KS):
        config = {"lags": LAG_CANDIDATES[lag_name], "target_transform": transform,
                  "spike_k": spike_k, "spike_window": SPIKE_WINDOW}
        features = build_features(frame, config)
        for model_name, make_model in model_candidates(transform).items():
            fold_scores = []
            for yr in VALIDATION_YEARS:
                train_mask, val_mask = masks(frame, config, yr)
                params = fit(frame, features, train_mask, config, make_model)
                fold_scores.append(metrics(y_true[val_mask],
                                           predict_rows(features, val_mask, params, config)))
            results.append({
                "lags": lag_name, "transform": transform, "spike_k": spike_k,
                "model": model_name, "n_features": features.shape[1],
                "val_r2": np.mean([s["r2"] for s in fold_scores]),
                "val_rmse": np.mean([s["rmse"] for s in fold_scores]),
                "val_mae": np.mean([s["mae"] for s in fold_scores]),
                **{f"r2_{yr}": s["r2"] for yr, s in zip(VALIDATION_YEARS, fold_scores)},
            })
        print(f"  evaluated lags={lag_name:16s} transform={transform:4s} spike_k={spike_k}")

    table = pd.DataFrame(results).sort_values("val_r2", ascending=False).reset_index(drop=True)
    table.to_csv(Path(__file__).resolve().parent / "model_selection_results.csv", index=False)
    pd.set_option("display.width", 200)
    print("\nTop 15 configurations by mean validation R2 (2020-2022):")
    print(table.head(15).round(4).to_string())

    best = table.iloc[0]
    config = {"lags": LAG_CANDIDATES[best["lags"]], "target_transform": best["transform"],
              "spike_k": None if pd.isna(best["spike_k"]) else float(best["spike_k"]),
              "spike_window": SPIKE_WINDOW}
    make_model = model_candidates(config["target_transform"])[best["model"]]
    features = build_features(frame, config)


    train_mask, test_mask = masks(frame, config, TEST_YEAR)
    params = fit(frame, features, train_mask, config, make_model)
    test = metrics(y_true[test_mask], predict_rows(features, test_mask, params, config))
    print(f"\nselected: lags={best['lags']} transform={best['transform']} "
          f"spike_k={config['spike_k']} model={best['model']}")
    print(f"held-out {TEST_YEAR}: R2={test['r2']:.4f} RMSE={test['rmse']:.4f} MAE={test['mae']:.4f}")


    all_mask = np.arange(len(frame)) >= max_lag(config)
    final = fit(frame, features, all_mask, config, make_model)
    model = {
        "description": "Linear ARX model for one-day-ahead chlorophyll-a",
        "estimator": best["model"],
        "config": config,
        "feature_names": list(features.columns),
        "coef": final["coef"],
        "intercept": final["intercept"],
        "mu": final["mu"],
        "sigma": final["sigma"],
        "smearing": final["smearing"],
        "fallback_value": float(np.nanmedian(y_true)),
        "validation": {"mean_r2_2020_2022": float(best["val_r2"]), f"test_{TEST_YEAR}": test},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as handle:
        pickle.dump(model, handle)
    print(f"saved {args.output} ({len(model['feature_names'])} features)")


    import my_model  # noqa: E402  (from best_model/)
    loaded = my_model.load_model(args.output)
    batch = predict_rows(features, all_mask, final, config)
    last_days = range(len(raw) - 10, len(raw))
    raw_sorted = raw.assign(**{DATE: pd.to_datetime(raw[DATE])}).sort_values(DATE).reset_index(drop=True)
    for i in last_days:
        history = raw_sorted.iloc[:i]
        day = raw_sorted.iloc[[i]].drop(columns=[TARGET])
        one = my_model.predict(loaded, history, day)[0]
        expected = batch[i - max_lag(config)]
        if not np.isclose(one, expected, rtol=1e-9, atol=1e-9):
            raise RuntimeError(f"predict() mismatch on {day[DATE].iloc[0].date()}: {one} vs {expected}")
    print("consistency check passed: predict() matches training-time features")


if __name__ == "__main__":
    main()
