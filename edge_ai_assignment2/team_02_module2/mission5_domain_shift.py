"""
mission5_domain_shift.py -- Mission 5: Laptop vs Board, Domain Shift & Nearest-Centroid Classification
Team 02 | AI255TBD Edge AI Module 2
"""

import csv
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

OUT_DIR = Path("results/mission5")
OUT_DIR.mkdir(parents=True, exist_ok=True)
SEED = 42


def save_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def clip_features_mean_std(X: np.ndarray) -> np.ndarray:
    """Aggregate (N, 49, 40) into (N, 80) per-clip mean + std as prescribed in README.md."""
    mu = X.mean(axis=1)
    sd = X.std(axis=1)
    return np.concatenate([mu, sd], axis=1)


def fit_nearest_centroid(X_tr: np.ndarray, y_tr: np.ndarray, n_classes: int):
    mu = X_tr.mean(axis=0)
    sd = X_tr.std(axis=0)
    sd[sd == 0] = 1.0
    Z_tr = (X_tr - mu) / sd
    centroids = np.vstack([Z_tr[y_tr == c].mean(axis=0) for c in range(n_classes)])
    return mu, sd, centroids


def predict_nearest_centroid(X_te: np.ndarray, mu: np.ndarray, sd: np.ndarray, centroids: np.ndarray):
    Z_te = (X_te - mu) / sd
    dists = np.linalg.norm(Z_te[:, None, :] - centroids[None, :, :], axis=2)
    return np.argmin(dists, axis=1), dists


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> dict:
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    acc = float(np.mean(y_true == y_pred))
    prec, rec, f1 = [], [], []
    for c in range(n_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        prec.append(p); rec.append(r); f1.append(f)
    return {
        "accuracy": round(acc, 4),
        "balanced_accuracy": round(float(np.mean(rec)), 4),
        "macro_precision": round(float(np.mean(prec)), 4),
        "macro_recall": round(float(np.mean(rec)), 4),
        "macro_f1": round(float(np.mean(f1)), 4),
        "cm": cm,
    }


def main():
    db = np.load("results/mission4/dataset_board.npz")
    dn = np.load("results/mission4/dataset_numpy.npz")

    Xb, yb, cids_b = db["X"], db["y"], db["clip_ids"]
    Xn, yn, cids_n = dn["X"], dn["y"], dn["clip_ids"]
    classes = list(db["classes"])
    n_classes = len(classes)

    assert Xb.shape == (144, 49, 40) and Xn.shape == (144, 49, 40)
    assert np.array_equal(yb, yn) and np.array_equal(cids_b, cids_n)
    assert np.isfinite(Xb).all() and np.isfinite(Xn).all()

    # 1. Domain shift analysis
    diff = np.abs(Xb - Xn)
    ds_rows = []
    for c_idx, cname in enumerate(classes):
        d = diff[yb == c_idx]
        ds_rows.append({
            "class": cname,
            "max_abs_err_db": round(float(np.max(d)), 6),
            "mean_abs_err_db": round(float(np.mean(d)), 6),
            "rmse_db": round(float(np.sqrt(np.mean(d ** 2))), 6),
        })
    save_csv(OUT_DIR / "domain_shift_metrics.csv", ds_rows)
    print("=== Domain Shift (Board vs NumPy) ===")
    print("Overall max:", round(float(np.max(diff)), 6), "mean:", round(float(np.mean(diff)), 6),
          "rmse:", round(float(np.sqrt(np.mean(diff ** 2))), 6))

    # 2. Stratified 2/3 train (16/class = 96), 1/3 test (8/class = 48)
    rng = np.random.default_rng(SEED)
    tr_idx, te_idx = [], []
    for c in range(n_classes):
        idx = np.flatnonzero(yb == c)
        perm = rng.permutation(idx)
        tr_idx.extend(perm[:16])
        te_idx.extend(perm[16:])
    tr_idx, te_idx = np.array(tr_idx), np.array(te_idx)
    assert len(set(tr_idx) & set(te_idx)) == 0

    Fb = clip_features_mean_std(Xb)
    Fn = clip_features_mean_std(Xn)

    experiments = [
        ("NumPy -> NumPy", Fn[tr_idx], yb[tr_idx], Fn[te_idx], yn[te_idx]),
        ("NumPy -> Board", Fn[tr_idx], yb[tr_idx], Fb[te_idx], yb[te_idx]),
        ("Board -> NumPy", Fb[tr_idx], yb[tr_idx], Fn[te_idx], yn[te_idx]),
        ("Board -> Board", Fb[tr_idx], yb[tr_idx], Fb[te_idx], yb[te_idx]),
    ]

    eval_rows = []
    pred_rows = []
    cms = {}
    for name, Xtr, ytr, Xte, yte in experiments:
        mu, sd, centroids = fit_nearest_centroid(Xtr, ytr, n_classes)
        ypred, dists = predict_nearest_centroid(Xte, mu, sd, centroids)
        m = compute_metrics(yte, ypred, n_classes)
        cms[name] = m.pop("cm")
        eval_rows.append({"experiment": name, "train_n": len(ytr), "test_n": len(yte), **m})
        print(name, m)
        for i, tid in enumerate(te_idx):
            pred_rows.append({
                "experiment": name,
                "clip_id": int(cids_b[tid]),
                "true_label": classes[int(yte[i])],
                "pred_label": classes[int(ypred[i])],
                "correct": bool(yte[i] == ypred[i]),
            })

    save_csv(OUT_DIR / "classifier_eval.csv", eval_rows)
    save_csv(OUT_DIR / "test_predictions.csv", pred_rows)

    # Plots
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([r["experiment"] for r in eval_rows], [r["accuracy"] * 100 for r in eval_rows],
           color=["tab:blue", "tab:cyan", "tab:orange", "tab:green"])
    ax.set_ylim(0, 105)
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Mission 5: Nearest-Centroid Cross-Domain Accuracy (48 Test Clips)")
    for i, r in enumerate(eval_rows):
        ax.text(i, r["accuracy"] * 100 + 2, f"{r['accuracy']*100:.1f}%", ha="center", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "cross_domain_accuracy.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, key in zip(axes, ["NumPy -> NumPy", "NumPy -> Board"]):
        cm = cms[key]
        ax.imshow(cm, cmap="Blues")
        ax.set_title(key)
        ax.set_xticks(range(n_classes)); ax.set_xticklabels(classes, rotation=45, ha="right")
        ax.set_yticks(range(n_classes)); ax.set_yticklabels(classes)
        for r_i in range(n_classes):
            for c_i in range(n_classes):
                ax.text(c_i, r_i, str(cm[r_i, c_i]), ha="center", va="center",
                        color="white" if cm[r_i, c_i] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "confusion_matrices.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
