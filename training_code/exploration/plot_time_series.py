"""Plot time-series exploration. Use --data to specify the released CSV or XLSX."""

from __future__ import annotations

from pathlib import Path
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "chlorophyll_student_2015_2023.csv"
FIG_DIR = Path(__file__).resolve().parent / "figures"
TARGET = "chlorophyll_a_mg_m3"

UNITS = {
    TARGET: "mg/m³",
    "sst_c": "°C",
    "par_umol_m2_s": "µmol/m²/s",
    "nitrate_umol_l": "µmol/L",
    "wind_speed_m_s": "m/s",
    "upwelling_index": "index",
    "mixed_layer_depth_m": "m",
    "salinity_psu": "PSU",
    "current_speed_m_s": "m/s",
    "river_discharge_index": "index",
    "cloud_fraction": "0–1",
    "surface_pressure_hpa": "hPa",
    "turbidity_ntu": "NTU",
}


def data_argument() -> Path:
    parser = argparse.ArgumentParser(description="Plot exploratory chlorophyll data analysis.")
    parser.add_argument("--data", type=Path, default=DATA)
    return parser.parse_args().data


def load(data_path: Path = DATA) -> pd.DataFrame:
    df = (pd.read_excel(data_path, sheet_name="daily_data")
          if data_path.suffix.lower() == ".xlsx" else pd.read_csv(data_path))
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def find_spikes(series: pd.Series, window: int = 31, k: float = 8.0) -> pd.Series:
    """Flag points far from a centred rolling median (robust MAD rule)."""
    median = series.rolling(window, center=True, min_periods=1).median()
    mad = (series - median).abs().rolling(window, center=True, min_periods=1).median()
    return (series - median).abs() > k * 1.4826 * mad


def plot_target(df: pd.DataFrame) -> None:
    y = df[TARGET]
    spikes = find_spikes(y)
    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.plot(df["date"], y, lw=0.6, color="tab:green", label="daily")
    ax.plot(df["date"], y.rolling(30, center=True).mean(), lw=1.8, color="black",
            label="30-day rolling mean")
    ax.scatter(df.loc[spikes, "date"], y[spikes], color="red", zorder=3, s=30,
               label=f"suspected spikes ({spikes.sum()})")
    ax.set_title("Chlorophyll-a (target), 2015–2023")
    ax.set_ylabel(UNITS[TARGET])
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "01_chlorophyll.png", dpi=120)
    plt.close(fig)
    print("Suspected chlorophyll spikes:")
    print(df.loc[spikes, ["date", TARGET]].to_string(index=False))


def plot_all(df: pd.DataFrame) -> None:
    cols = [c for c in df.columns if c != "date"]
    fig, axes = plt.subplots(len(cols), 1, figsize=(14, 2.1 * len(cols)), sharex=True)
    for ax, col in zip(axes, cols):
        ax.plot(df["date"], df[col], lw=0.4, color="tab:blue", alpha=0.7)
        ax.plot(df["date"], df[col].rolling(30, center=True).mean(), lw=1.4, color="black")
        ax.set_title(f"{col}  [{UNITS[col]}]", fontsize=10, loc="left")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_all_variables.png", dpi=100)
    plt.close(fig)


def plot_seasonal_cycle(df: pd.DataFrame) -> None:
    """Average of each variable per day of year, standardized so all fit on one scale."""
    doy = df["date"].dt.dayofyear
    cols = [c for c in df.columns if c != "date"]
    clim = df[cols].groupby(doy).mean().rolling(15, center=True, min_periods=1).mean()
    z = (clim - clim.mean()) / clim.std()
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
    corr = df[cols].corr()[TARGET].drop(TARGET)
    pos = list(corr[corr >= 0].index)
    neg = list(corr[corr < 0].index)
    for ax, group, title in [(axes[0], pos, "Positively correlated with chlorophyll"),
                             (axes[1], neg, "Negatively correlated with chlorophyll")]:
        ax.plot(z.index, z[TARGET], lw=3, color="tab:green", label=TARGET)
        for col in group:
            ax.plot(z.index, z[col], lw=1.2, label=col)
        ax.set_title(title)
        ax.set_xlabel("day of year")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("standardized mean (z-score)")
    fig.suptitle("Average seasonal cycle (2015–2023)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "03_seasonal_cycle.png", dpi=110)
    plt.close(fig)


def plot_one_year(df: pd.DataFrame, year: int = 2023) -> None:
    """Zoom on one year to see day-to-day dynamics hidden in the 9-year view."""
    sub = df[df["date"].dt.year == year]
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(sub["date"], sub[TARGET], lw=1, color="tab:green", marker=".", ms=3)
    ax.set_title(f"Chlorophyll-a during {year} (daily)")
    ax.set_ylabel(UNITS[TARGET])
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "04_chlorophyll_one_year.png", dpi=120)
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = load(data_argument())
    print(f"{len(df)} rows, {df['date'].min().date()} to {df['date'].max().date()}")
    plot_target(df)
    plot_all(df)
    plot_seasonal_cycle(df)
    plot_one_year(df)
    print(f"Figures saved to {FIG_DIR}")


if __name__ == "__main__":
    main()
