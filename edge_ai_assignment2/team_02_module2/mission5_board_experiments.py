"""
mission5_board_experiments.py -- Mission 5: Actual ESP32 Board Streams for int8 & 8 kHz
Team 02 | AI255TBD Edge AI Module 2

Streams all 144 clips on COM5 (921600 baud) to collect actual board-computed features for:
  1. int8, S=1.0, 16 kHz
  2. int8, S=0.25, 16 kHz
  3. int16, S=1.0, 8 kHz, antialias=False (compared against filtered 8 kHz laptop audio)

Validates counts/offsets/CRCs, computes overall/per-band/per-class discrepancies,
generates heatmaps, and evaluates nearest-centroid (96/48 split, seed=42).
"""

import csv
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from board_link import Board
from team_audio import CLASSES, clip_list, get_clip
from mission2_sampling_storage import quantise, pack
from mission3_features import features as compute_features

TEAM = 2
COM_PORT = "COM5"
BAUD = 921600
SEED = 42
MELS = 40
OUT_DIR = Path("results/mission5")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def save_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def clip_features_mean_std(X: np.ndarray) -> np.ndarray:
    return np.concatenate([X.mean(axis=1), X.std(axis=1)], axis=1)


def fit_nc(Xtr: np.ndarray, ytr: np.ndarray, n_classes: int):
    mu = Xtr.mean(axis=0)
    sd = Xtr.std(axis=0)
    sd[sd == 0] = 1.0
    Ztr = (Xtr - mu) / sd
    centroids = np.vstack([Ztr[ytr == c].mean(axis=0) for c in range(n_classes)])
    return mu, sd, centroids


def predict_nc(Xte: np.ndarray, mu, sd, centroids):
    Zte = (Xte - mu) / sd
    dists = np.linalg.norm(Zte[:, None, :] - centroids[None, :, :], axis=2)
    return np.argmin(dists, axis=1)


def compute_metrics(ytrue, ypred, n_classes):
    acc = float(np.mean(ytrue == ypred))
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(ytrue, ypred):
        cm[t, p] += 1
    rec = [cm[c, c] / max(cm[c].sum(), 1) for c in range(n_classes)]
    return {"accuracy": round(acc, 4), "balanced_accuracy": round(float(np.mean(rec)), 4)}


def stream_and_extract(name: str, fs: int, fmt: str, scale: float, antialias_board: bool,
                       antialias_laptop: bool, win: int, hop: int, nfft: int, spp: int):
    """Stream all 144 clips to ESP32, validate, and return (X_board, X_numpy_quant, X_laptop_ref, stats)."""
    print(f"\n=== Streaming {name} (fs={fs}, fmt={fmt}, S={scale}, aa_board={antialias_board}) ===")
    clips = clip_list(TEAM)
    x_board_src = np.concatenate([get_clip(TEAM, cid, fs=fs, antialias=antialias_board) for cid, _ in clips])
    _, xhat_all, _ = quantise(x_board_src, fmt=fmt, scale=scale)
    pl_all = pack(x_board_src, fmt=fmt, scale=scale)
    ref_quant = compute_features(xhat_all, fs=fs, win=win, hop=hop, nfft=nfft, mels=MELS)

    # Original laptop reference (for 8 kHz, filtered 8 kHz laptop audio)
    X_laptop = np.zeros((144, 49, MELS), dtype=np.float32)
    for i, (cid, _) in enumerate(clips):
        x_lap = get_clip(TEAM, cid, fs=fs, antialias=antialias_laptop)
        F_lap = compute_features(x_lap, fs=fs, win=win, hop=hop, nfft=nfft, mels=MELS)
        X_laptop[i] = F_lap[:49]

    b = Board(COM_PORT, baud=BAUD)
    try:
        r = b.stream(pl_all, fs=fs, fmt=fmt, scale=scale, win=win, hop=hop,
                     nfft=nfft, mels=MELS, mfcc=0, feat=True, store="none",
                     speed=1.0, samples_per_packet=spp)
    finally:
        b.close()
    time.sleep(0.4)

    brd = r["board"] or {}
    got_feat = r["features"]
    fsamp = np.asarray(r["first_sample"], dtype=np.int64)
    fnums = np.asarray(r["frame_numbers"], dtype=np.int64)

    assert brd.get("samples", 0) == len(x_board_src), f"Sample mismatch: {brd.get('samples')} != {len(x_board_src)}"
    assert len(got_feat) == r["expected_features"], f"Frame mismatch: {len(got_feat)} != {r['expected_features']}"
    assert brd.get("bad_crc", 0) == 0 and brd.get("seq_gaps", 0) == 0
    assert brd.get("uart_errors", 0) == 0 and r.get("bad_crc_back", 0) == 0
    assert np.isfinite(got_feat).all(), "Non-finite board features!"

    X_board = np.zeros((144, 49, MELS), dtype=np.float32)
    X_numpy_q = np.zeros((144, 49, MELS), dtype=np.float32)
    occupied = np.zeros((144, 49), dtype=bool)
    for i, (fn, s0) in enumerate(zip(fnums, fsamp)):
        assert int(s0) == int(fn) * hop, f"Offset mismatch: {s0} != {fn * hop}"
        c_start = int(s0 // fs)
        c_end = int((s0 + win - 1) // fs)
        if c_start != c_end:
            continue
        fic = int((s0 % fs) // hop)
        occupied[c_start, fic] = True
        X_board[c_start, fic] = got_feat[i]
        X_numpy_q[c_start, fic] = ref_quant[fn]

    assert occupied.all(), "Unpopulated feature slots!"
    diff_quant = np.abs(X_board - X_numpy_q)
    diff_lap = np.abs(X_board - X_laptop)

    stats = {
        "config": name,
        "fs": fs,
        "fmt": fmt,
        "scale": scale,
        "antialias_board": antialias_board,
        "samples_received": brd.get("samples", 0),
        "received_features": len(got_feat),
        "bad_crc": brd.get("bad_crc", 0),
        "seq_gaps": brd.get("seq_gaps", 0),
        "compute_median_us": round(float(np.median(r["compute_us"])), 1),
        "board_vs_quant_numpy_max_db": round(float(diff_quant.max()), 6),
        "board_vs_quant_numpy_mean_db": round(float(diff_quant.mean()), 6),
        "board_vs_laptop_max_db": round(float(diff_lap.max()), 6),
        "board_vs_laptop_mean_db": round(float(diff_lap.mean()), 6),
        "board_vs_laptop_rmse_db": round(float(np.sqrt(np.mean(diff_lap ** 2))), 6),
    }
    print("  Stats:", stats)
    return X_board, X_numpy_q, X_laptop, stats


def main():
    clips = clip_list(TEAM)
    y = np.array([CLASSES.index(lab) for _, lab in clips], dtype=np.int64)
    n_classes = len(CLASSES)

    rng = np.random.default_rng(SEED)
    tr_idx, te_idx = [], []
    for c in range(n_classes):
        idx = np.flatnonzero(y == c)
        perm = rng.permutation(idx)
        tr_idx.extend(perm[:16])
        te_idx.extend(perm[16:])
    tr_idx, te_idx = np.array(tr_idx), np.array(te_idx)

    configs = [
        ("int8_S1_16k", 16000, "i8", 1.0, True, True, 480, 320, 512, 320),
        ("int8_S025_16k", 16000, "i8", 0.25, True, True, 480, 320, 512, 320),
        ("int16_S1_8k_noaa", 8000, "i16", 1.0, False, True, 240, 160, 256, 160),
    ]

    disc_rows = []
    class_rows = []
    band_rows = []
    cls_rows = []
    heatmaps = []

    for name, fs, fmt, scale, aa_b, aa_l, win, hop, nfft, spp in configs:
        Xb, Xnq, Xlap, st = stream_and_extract(name, fs, fmt, scale, aa_b, aa_l, win, hop, nfft, spp)
        disc_rows.append(st)

        diff_lap = np.abs(Xb - Xlap)
        heatmaps.append((name, diff_lap.mean(axis=0)))

        for c_idx, cname in enumerate(CLASSES):
            d = diff_lap[y == c_idx]
            class_rows.append({
                "config": name, "class": cname,
                "max_abs_err_db": round(float(d.max()), 6),
                "mean_abs_err_db": round(float(d.mean()), 6),
                "rmse_db": round(float(np.sqrt(np.mean(d ** 2))), 6),
            })
        for m in range(MELS):
            d = diff_lap[:, :, m]
            band_rows.append({
                "config": name, "mel_band": m,
                "max_abs_err_db": round(float(d.max()), 6),
                "mean_abs_err_db": round(float(d.mean()), 6),
            })

        Flap = clip_features_mean_std(Xlap)
        Fb = clip_features_mean_std(Xb)
        mu, sd, centroids = fit_nc(Flap[tr_idx], y[tr_idx], n_classes)
        m_lap = compute_metrics(y[te_idx], predict_nc(Flap[te_idx], mu, sd, centroids), n_classes)
        m_brd = compute_metrics(y[te_idx], predict_nc(Fb[te_idx], mu, sd, centroids), n_classes)
        cls_rows.append({"config": name, "experiment": "Laptop -> Laptop", **m_lap})
        cls_rows.append({"config": name, "experiment": "Laptop -> Board", **m_brd})

    save_csv(OUT_DIR / "board_experiments_discrepancy.csv", disc_rows)
    save_csv(OUT_DIR / "board_experiments_per_class.csv", class_rows)
    save_csv(OUT_DIR / "board_experiments_per_band.csv", band_rows)
    save_csv(OUT_DIR / "board_experiments_classifier.csv", cls_rows)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle("Mission 5: Actual ESP32 Board vs Laptop Reference — Mean |Δ| (dB)", fontsize=11)
    for ax, (name, hmap) in zip(axes, heatmaps):
        im = ax.imshow(hmap.T, aspect="auto", origin="lower", cmap="hot")
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("Frame")
        ax.set_ylabel("Mel Band")
        plt.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "board_experiments_heatmaps.png", dpi=150)
    plt.close(fig)
    print("\nSaved board_experiments_*.csv and board_experiments_heatmaps.png")


if __name__ == "__main__":
    main()
