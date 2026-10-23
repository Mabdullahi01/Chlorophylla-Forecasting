"""Explore target and exogenous lags. Use --data for the released CSV or XLSX.

Exploration uses 2015-2023, including the later evaluation year."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_time_series import FIG_DIR, TARGET, data_argument, find_spikes, load

MAX_LAG = 20
AR_ORDER = 14


def despike_target(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    y = out[TARGET]
    spikes = find_spikes(y)
    out.loc[spikes, TARGET] = y.rolling(31, center=True, min_periods=1).median()[spikes]
    print(f"replaced {spikes.sum()} chlorophyll spikes with the rolling median")
    return out


def anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """Subtract a smoothed day-of-year mean from every variable."""
    doy = df["date"].dt.dayofyear.clip(upper=365)
    cols = [c for c in df.columns if c != "date"]
    clim = df[cols].groupby(doy).mean()
    # circular smoothing so 31 Dec and 1 Jan are neighbours
    padded = pd.concat([clim.tail(15), clim, clim.head(15)])
    clim = padded.rolling(31, center=True).mean().iloc[15:-15]
    return df[cols] - clim.loc[doy].to_numpy()


def lagged_corr(y: np.ndarray, x: np.ndarray, lag: int) -> float:
    """corr(y(t), x(t - lag))"""
    if lag > 0:
        y, x = y[lag:], x[:-lag]
    ok = np.isfinite(y) & np.isfinite(x)
    return float(np.corrcoef(y[ok], x[ok])[0, 1])


def pacf(y: np.ndarray, max_lag: int) -> list[float]:
    """Partial autocorrelation via successive least-squares AR fits."""
    values = []
    for k in range(1, max_lag + 1):
        X = np.column_stack([y[k - i:len(y) - i] for i in range(1, k + 1)])
        X = np.column_stack([np.ones(len(X)), X])
        coef, *_ = np.linalg.lstsq(X, y[k:], rcond=None)
        values.append(coef[-1])
    return values


def ar_residual(y: np.ndarray, order: int) -> np.ndarray:
    """Residual of y(t) after regressing on y(t-1..t-order); NaN for the first rows."""
    X = np.column_stack([y[order - i:len(y) - i] for i in range(1, order + 1)])
    X = np.column_stack([np.ones(len(X)), X])
    coef, *_ = np.linalg.lstsq(X, y[order:], rcond=None)
    resid = np.full(len(y), np.nan)
    resid[order:] = y[order:] - X @ coef
    return resid


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = despike_target(load(data_argument()))
    anom = anomalies(df)
    y = anom[TARGET].to_numpy()
    n = len(y)
    band = 1.96 / np.sqrt(n)  # approx. 95% band for "no correlation"


    lags = np.arange(1, 31)
    acf = [lagged_corr(y, y, k) for k in lags]
    pac = pacf(y, len(lags))
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    for ax, vals, title in [(axes[0], acf, "ACF of chlorophyll anomaly"),
                            (axes[1], pac, "PACF of chlorophyll anomaly")]:
        ax.bar(lags, vals, color="tab:green")
        ax.axhspan(-band, band, color="grey", alpha=0.25, label="95% no-correlation band")
        ax.set_title(title)
        ax.set_xlabel("lag (days)")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "05_target_acf_pacf.png", dpi=120)
    plt.close(fig)
    print("\nTarget PACF (lag: value), significant ones marked *")
    print("  " + "  ".join(f"{k}:{v:+.2f}{'*' if abs(v) > band else ''}"
                           for k, v in zip(lags, pac)))


    resid = ar_residual(y, AR_ORDER)
    exog = [c for c in anom.columns if c != TARGET]
    xlags = np.arange(0, MAX_LAG + 1)
    fig, axes = plt.subplots(4, 3, figsize=(16, 13), sharex=True)
    rows = []
    for ax, col in zip(axes.flat, exog):
        x = anom[col].to_numpy()
        c_anom = np.array([lagged_corr(y, x, j) for j in xlags])
        c_res = np.array([lagged_corr(resid, x, j) for j in xlags])
        ax.bar(xlags - 0.2, c_anom, width=0.4, color="tab:blue", label="vs chlorophyll anomaly")
        ax.bar(xlags + 0.2, c_res, width=0.4, color="tab:orange", label="vs AR residual")
        ax.axhspan(-band, band, color="grey", alpha=0.25)
        ax.axhline(0, color="black", lw=0.6)
        ax.set_title(col, fontsize=10)
        ax.grid(alpha=0.3)
        sig = [int(j) for j in xlags if abs(c_res[j]) > band]
        rows.append({"variable": col,
                     "peak_lag_anom": int(np.argmax(np.abs(c_anom))),
                     "peak_corr_anom": round(float(c_anom[np.argmax(np.abs(c_anom))]), 3),
                     "peak_lag_resid": int(np.argmax(np.abs(c_res))),
                     "peak_corr_resid": round(float(c_res[np.argmax(np.abs(c_res))]), 3),
                     "significant_resid_lags": sig})
    for ax in axes[-1]:
        ax.set_xlabel("lag j (days): x(t - j)")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("corr( chlorophyll(t), x(t - j) ) after removing seasonality; "
                 "grey = 95% no-correlation band", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "06_exog_lag_correlations.png", dpi=110)
    plt.close(fig)

    table = pd.DataFrame(rows).sort_values("peak_corr_resid", key=np.abs, ascending=False)
    print("\nExogenous lag summary (sorted by strength beyond chlorophyll's own past)")
    print(table.to_string(index=False))


    raw_corr = df.drop(columns=["date", TARGET]).corr()
    anom_corr = anom.drop(columns=[TARGET]).corr()
    fig, axes = plt.subplots(1, 2, figsize=(17, 7))
    for ax, mat, title in [(axes[0], raw_corr, "raw values"),
                           (axes[1], anom_corr, "seasonal anomalies")]:
        im = ax.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(mat)), mat.columns, rotation=60, ha="right", fontsize=8)
        ax.set_yticks(range(len(mat)), mat.columns, fontsize=8)
        for i in range(len(mat)):
            for j in range(len(mat)):
                ax.text(j, i, f"{mat.iat[i, j]:.1f}", ha="center", va="center", fontsize=6)
        ax.set_title(f"Predictor correlations: {title}")
    fig.colorbar(im, ax=axes, shrink=0.8)
    fig.savefig(FIG_DIR / "07_predictor_correlations.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigures saved to {FIG_DIR}")


if __name__ == "__main__":
    main()
