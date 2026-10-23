"""Plot the existing model search and reproduce the selected-model baseline comparison.

Run with --data PATH_TO_RELEASED_CSV_OR_XLSX. Requires matplotlib for plotting.
Does not change the search results or the submitted model.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import train as t

HERE = Path(__file__).resolve().parent
FAMILIES = ["OLS", "Ridge", "LASSO"]
COLORS = ["#3973ac", "#dc8744", "#208b79"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--results", type=Path, default=HERE.parent / "model_selection_results.csv")
    parser.add_argument("--output-dir", type=Path, default=HERE)
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(args.results).sort_values("val_r2", ascending=False).reset_index(drop=True)
    table["family"] = table["model"].str.split("_").str[0].map({"ols": "OLS", "ridge": "Ridge", "lasso": "LASSO"})
    raw = (pd.read_excel(args.data, sheet_name="daily_data") if args.data.suffix.lower() == ".xlsx"
           else pd.read_csv(args.data))
    frame = t.prepare_raw(raw)
    years = t.VALIDATION_YEARS + [t.TEST_YEAR]
    y = frame[t.TARGET].to_numpy()
    persistence = frame[t.TARGET].shift(1).to_numpy()
    baseline = {}
    for year in years:
        mask = frame[t.DATE].dt.year.to_numpy() == year
        baseline[year] = t.metrics(y[mask], persistence[mask])
    baseline_mean = {k: np.mean([baseline[yr][k] for yr in t.VALIDATION_YEARS])
                     for k in ["r2", "rmse", "mae"]}

    # R2 is a score (higher is better), not an error. Lag sets are categories.
    lags = list(t.LAG_CANDIDATES)
    lag_labels = ["Provided", "Explored", "Short", "No weak", "Long"]
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.facecolor": "white"})
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharey=True)
    summaries = []
    for ax, (transform, spike) in zip(axes.flat, [("none", None), ("none", 8.0), ("log", None), ("log", 8.0)]):
        sub = table[(table["transform"] == transform) &
                    (table["spike_k"].isna() if spike is None else table["spike_k"].eq(spike))]
        for family, color in zip(FAMILIES, COLORS):
            best = sub[sub["family"] == family].drop_duplicates("lags").set_index("lags").loc[lags]
            ax.plot(lag_labels, best["val_r2"], marker="o", label=family, color=color)
            summaries.append(best.reset_index())
        ax.axhline(baseline_mean["r2"], color="#555555", ls="--", label="Persistence")
        ax.set_title(f"Target: {transform} | Despiking: {'off' if spike is None else 'k=8'}")
        ax.set_ylabel("Mean validation R² (higher is better)")
        ax.grid(axis="y", alpha=.2)
    axes[0, 0].legend(ncol=2)
    fig.suptitle("Lag combinations versus fitting method", fontsize=17)
    fig.text(.5, .025, "2020–2022 expanding-window validation; best alpha within each family and lag set.\n"
             "Lines connect categorical lag sets, not a numerical lag-length sweep.", ha="center")
    fig.tight_layout(rect=(0, .07, 1, .95))
    save(fig, out, "01_r2_by_lag_combination")
    pd.concat(summaries).to_csv(out / "lag_family_comparison.csv", index=False)

    # Show every fitting option at the selected preprocessing, without hiding alpha.
    winner = table.iloc[0]
    selected = table[(table["transform"] == winner["transform"]) &
                     (table["spike_k"].isna() if pd.isna(winner["spike_k"]) else table["spike_k"].eq(winner["spike_k"]))]
    model_order = list(t.model_candidates(winner["transform"]))
    matrix = selected.pivot(index="model", columns="lags", values="val_r2").loc[model_order, lags]
    fig, ax = plt.subplots(figsize=(11, 8))
    im = ax.imshow(matrix.to_numpy(), cmap="YlGnBu", aspect="auto")
    ax.set_xticks(range(len(lags)), lag_labels)
    ax.set_yticks(range(len(model_order)), [s.upper().replace("_", " α=") for s in model_order])
    midpoint = (matrix.to_numpy().min() + matrix.to_numpy().max()) / 2
    for i in range(len(model_order)):
        for j in range(len(lags)):
            v = matrix.iloc[i, j]
            ax.text(j, i, f"{v:.4f}", ha="center", va="center", color="white" if v > midpoint else "black")
    ax.set_title(f"Every fitting option at the selected preprocessing\n"
                 f"Target: {winner['transform']} | spike k: {winner['spike_k']} | mean validation R²", pad=15)
    fig.colorbar(im, ax=ax, label="R² (higher is better)")
    fig.text(.5, .025, f"Persistence baseline: {baseline_mean['r2']:.4f}. Alpha is the regularization strength; OLS has no penalty.", ha="center")
    fig.tight_layout(rect=(0, .06, 1, 1))
    save(fig, out, "02_all_fitting_options")

    # Compare the best validation configuration of each family, and a true simple baseline.
    family_best = table.drop_duplicates("family").set_index("family").loc[FAMILIES].reset_index()
    family_best["r2_gain_over_baseline"] = family_best["val_r2"] - baseline_mean["r2"]
    family_best["rmse_reduction_pct"] = 100 * (1 - family_best["val_rmse"] / baseline_mean["rmse"])
    family_best["mae_reduction_pct"] = 100 * (1 - family_best["val_mae"] / baseline_mean["mae"])
    base_row = {"family": "Persistence", "model": "previous_day", "val_r2": baseline_mean["r2"],
                "val_rmse": baseline_mean["rmse"], "val_mae": baseline_mean["mae"]}
    pd.concat([pd.DataFrame([base_row]), family_best], ignore_index=True).to_csv(out / "fitting_method_comparison.csv", index=False)

    config = {"lags": t.LAG_CANDIDATES[winner["lags"]], "target_transform": winner["transform"],
              "spike_k": None if pd.isna(winner["spike_k"]) else float(winner["spike_k"]), "spike_window": t.SPIKE_WINDOW}
    features = t.build_features(frame, config)
    make_model = t.model_candidates(config["target_transform"])[winner["model"]]
    rows = []
    for year in years:
        train_mask, eval_mask = t.masks(frame, config, year)
        params = t.fit(frame, features, train_mask, config, make_model)
        score = t.metrics(y[eval_mask], t.predict_rows(features, eval_mask, params, config))
        if year in t.VALIDATION_YEARS and not np.isclose(score["r2"], winner[f"r2_{year}"], atol=1e-9):
            raise ValueError("Reproduced score differs from search CSV; check data and results paths")
        for name, result in [("Persistence", baseline[year]), ("Selected LASSO" if winner["family"] == "LASSO" else "Selected " + winner["family"], score)]:
            rows.append({"year": year, "model": name, **result})
    yearly = pd.DataFrame(rows)
    yearly.to_csv(out / "selected_vs_baseline_by_year.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    values = [baseline_mean["r2"]] + family_best["val_r2"].tolist()
    bars = axes[0].bar(["Persistence"] + FAMILIES, values, color=["#9a9a9a"] + COLORS)
    axes[0].bar_label(bars, fmt="%.4f", padding=4)
    axes[0].set_title("Best configuration per fitting method\nMean validation R², 2020–2022")
    x = np.arange(len(years))
    for offset, (name, color) in zip([-.18, .18], [("Persistence", "#9a9a9a"), (rows[-1]["model"], COLORS[2])]):
        vals = yearly[yearly["model"] == name]["r2"]
        bars = axes[1].bar(x + offset, vals, .36, label=name, color=color)
        axes[1].bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
    axes[1].set_xticks(x, [str(y) for y in years])
    axes[1].set_title("Selected configuration versus persistence\nEach year predicted using a fit on earlier years")
    axes[1].legend(loc="lower left")
    for ax in axes:
        ax.set_ylabel("R² (higher is better)")
        ax.set_ylim(min(0, min(yearly["r2"]), min(values)) - .03, max(max(values), max(yearly["r2"])) + .1)
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
    fig.text(.5, .025, "2023 is excluded from hyperparameter selection. Earlier exploratory lag analysis used 2023,\n"
             "so its score should not be described as a completely untouched holdout.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .09, 1, 1))
    save(fig, out, "03_models_vs_baseline")
    lag_rows = [{"lag_set": name, "variable": variable, "lags": ",".join(map(str, indices))}
                for name, spec in t.LAG_CANDIDATES.items() for variable, indices in spec.items()]
    pd.DataFrame(lag_rows).to_csv(out / "lag_definitions.csv", index=False)
    print(family_best[["family", "model", "lags", "transform", "spike_k", "val_r2", "val_rmse", "val_mae"]].to_string(index=False))
    print(yearly.to_string(index=False))
    print(f"Saved plots and comparison tables to {out}")


def save(fig, out, name):
    fig.savefig(out / (name + ".png"), dpi=180)
    fig.savefig(out / (name + ".svg"))
    plt.close(fig)


if __name__ == "__main__":
    main()
