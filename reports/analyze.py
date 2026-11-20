"""Run from the project root with python -m reports.analyze."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from app.sensors import FEATURES
from training_data import DATA, baselines, clean, split

OUT = Path(__file__).parent


def analyze():
    raw = pd.read_csv(DATA)
    good, invalid = clean(raw)
    train, _ = split(good)
    fit, val = split(train, 31)
    mean, linear = baselines(fit, val)
    print("Types", raw.dtypes, "Missing", raw.isna().sum(), "Duplicates", raw.duplicated().sum())
    print("Rejected per column", invalid.sum(), "Valid", len(good))
    print("Raw", raw.describe().T.to_string(), "Clean", good.describe().T.to_string())
    print("Development Pearson correlations", train.corr().round(2).to_string())
    print(
        "Development baseline MAEs",
        np.abs(mean - val.days_left).mean(),
        np.abs(linear - val.days_left).mean(),
    )
    print(
        pd.DataFrame({"target_bin": pd.qcut(val.days_left, 4), "residual": linear - val.days_left})
        .groupby("target_bin", observed=True)
        .residual.agg(["mean", lambda x: x.abs().mean()])
    )
    fig, axes = plt.subplots(2, 3, figsize=(11, 6), constrained_layout=True)
    for ax, col in zip(axes.flat, raw):
        ax.hist(raw[col], bins=40, color="#306684")
        ax.set_title(col)
        ax.set_ylabel("Rows")
    fig.savefig(OUT / "histograms.png", dpi=135)
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.5), constrained_layout=True)
    for ax, col in zip(axes.flat, FEATURES):
        ax.scatter(train[col], train.days_left, s=3, alpha=0.20)
        bins = pd.qcut(train[col], 12, duplicates="drop")
        groups = train.groupby(bins, observed=True)[[col, "days_left"]].mean()
        ax.plot(groups[col], groups.days_left, color="darkorange", lw=2)
        ax.set(xlabel=col, ylabel="Days left")
    ax = axes.flat[-1]
    ax.scatter(val.days_left, linear, s=6, alpha=0.45)
    ax.plot([0, 280], [0, 280], color="darkorange")
    ax.set(xlabel="True days", ylabel="Linear prediction", title="Development validation")
    fig.savefig(OUT / "relationships.png", dpi=135)
    plt.close(fig)


if __name__ == "__main__":
    analyze()
