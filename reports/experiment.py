"""Three-fold selection entirely inside the development partition."""

import json
from dataclasses import asdict, replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from train import Options, fit, score
from training_data import DATA, clean, split

OUT = Path(__file__).parent


def main():
    dev, _ = split(clean(pd.read_csv(DATA))[0])
    folds = np.array_split(np.random.default_rng(31).permutation(len(dev)), 3)
    rows, options = [], {}

    def evaluate(name, option):
        options[name] = option
        for i, test_ids in enumerate(folds):
            train_ids = np.concatenate([ids for j, ids in enumerate(folds) if j != i])
            training, validation = dev.iloc[train_ids], dev.iloc[test_ids]
            net, mu, sd, result = fit(training, replace(option, seed=option.seed + i), validation)
            result.update(
                candidate=name,
                fold=i,
                train_mae=score(net, mu, sd, training),
                val_mae=score(net, mu, sd, validation),
            )
            result["gap"] = result["val_mae"] - result["train_mae"]
            rows.append(result)
        pd.DataFrame(rows).to_csv(OUT / "experiments.csv", index=False)
        print(name, round(np.mean([r["val_mae"] for r in rows[-3:]]), 3), flush=True)

    for depth in (1, 2):
        for w in (4, 8, 16, 32, 64):
            evaluate(f"width{w}x{depth}", Options(hidden=(w,) * depth))
    grid = pd.DataFrame(rows)
    leaders = grid.groupby("candidate").val_mae.mean().nsmallest(2).index
    for name in leaders:
        opt = options[name]
        for label, changes in [
            ("long", {"epochs": 3200, "lr": 0.003}),
            ("huber", {"epochs": 3200, "lr": 0.003, "loss": "huber", "decay": 1e-5}),
            ("restart", {"epochs": 2000, "lr": 0.009, "seed": 71}),
        ]:
            evaluate(f"{name}-{label}", replace(opt, **changes))
    evaluate("unnormalized32x2", Options(normalize=False))
    table = pd.DataFrame(rows)
    summary = table.groupby("candidate").agg(mae=("val_mae", "mean"), params=("params", "first"))
    # Use the smaller model within 0.05 day (72 minutes) of the best mean.
    champion = (
        summary[summary.mae <= summary.mae.min() + 0.05].sort_values(["params", "mae"]).index[0]
    )
    selected = table[table.candidate == champion]
    recipe = asdict(replace(options[champion], epochs=int(selected.best_epoch.median())))
    (OUT / "recipe.json").write_text(json.dumps(recipe, indent=2) + "\n")
    print("Selected", champion, recipe, flush=True)
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    for depth in (1, 2):
        by_width = grid[grid.candidate.str.endswith(f"x{depth}")].groupby("params").val_mae
        stats = by_width.agg(["min", "max", "mean"])
        ax.plot(stats.index, stats["mean"], "o-", label=f"{depth} hidden layer(s)")
        ax.fill_between(stats.index, stats["min"], stats["max"], alpha=0.15)
    ax.set(xscale="log", xlabel="Trainable parameters (log scale)", ylabel="Validation MAE (days)")
    ax.legend()
    fig.savefig(OUT / "architecture.png", dpi=150)


if __name__ == "__main__":
    main()
