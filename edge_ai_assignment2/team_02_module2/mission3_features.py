"""
mission3_features.py -- Mission 3: Feature Extraction (NumPy reference, baseline & optimized benchmarks)
Team 02 | AI255TBD Edge AI Module 2
"""

import argparse
import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from board_link import Board
from team_audio import get_clip
from mission2_sampling_storage import quantise, pack

TEAM = 2
COM_PORT = "COM5"
USB_PORT = "COM10"
OUT_DIR = Path("results/mission3")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def features(x: np.ndarray, fs: int = 16000, win: int = 480, hop: int = 320,
             nfft: int = 512, mels: int = 40, mfcc: int = 0) -> np.ndarray:
    """NumPy reference matching README.md Section 5 and features.h."""
    x = np.asarray(x, dtype=np.float64)
    n_frames = (len(x) - win) // hop + 1
    if n_frames <= 0:
        return np.zeros((0, mfcc or mels), dtype=np.float32)

    w = np.hanning(win)
    nb = nfft // 2 + 1

    def hz2mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel2hz(m):
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    edges = mel2hz(np.linspace(hz2mel(20.0), hz2mel(fs / 2.0), mels + 2))
    freqs = np.arange(nb) * (fs / nfft)
    melw = np.zeros((mels, nb), dtype=np.float64)
    for m in range(mels):
        lo, ce, hi = edges[m], edges[m + 1], edges[m + 2]
        up = (freqs > lo) & (freqs <= ce)
        dn = (freqs > ce) & (freqs < hi)
        melw[m, up] = (freqs[up] - lo) / (ce - lo)
        melw[m, dn] = (hi - freqs[dn]) / (hi - ce)

    out = np.empty((n_frames, mfcc or mels), dtype=np.float32)
    if mfcc:
        q = np.arange(mfcc)[:, None]
        m_idx = np.arange(mels)[None, :]
        scale = np.where(q == 0, np.sqrt(1.0 / mels), np.sqrt(2.0 / mels))
        dct = scale * np.cos(np.pi * q * (2 * m_idx + 1) / (2.0 * mels))

    for k in range(n_frames):
        s = k * hop
        frame = x[s:s + win] * w
        spec = np.fft.rfft(frame, n=nfft)
        pw = (spec.real ** 2 + spec.imag ** 2) / nfft
        melv = 10.0 * np.log10(np.maximum(melw @ pw, 1e-10))
        out[k] = (dct @ melv) if mfcc else melv
    return out


def save_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def run_benchmark(port: str, baud: int, clips_sec: int, win: int, hop: int,
                  nfft: int, mels: int, mfcc: int, speed: float, label: str, run_idx: int = 1) -> tuple[dict, np.ndarray, np.ndarray]:
    x = np.concatenate([get_clip(TEAM, i, fs=16000, antialias=True) for i in range(clips_sec)])
    _, xhat, _ = quantise(x, fmt="i16", scale=1.0)
    pl = pack(x, fmt="i16", scale=1.0)
    ref = features(xhat, fs=16000, win=win, hop=hop, nfft=nfft, mels=mels, mfcc=mfcc)

    b = Board(port, baud=baud)
    try:
        r = b.stream(pl, fs=16000, fmt="i16", scale=1.0, win=win, hop=hop,
                     nfft=nfft, mels=mels, mfcc=mfcc, feat=True, store="none",
                     speed=speed, samples_per_packet=320)
    finally:
        b.close()

    brd = r["board"] or {}
    cu = np.asarray(r["compute_us"], dtype=np.float64)
    lag = np.asarray(r["lag_s"], dtype=np.float64)
    got_feat = r["features"]
    fnums = np.asarray(r["frame_numbers"], dtype=np.int64)
    fsamp = np.asarray(r["first_sample"], dtype=np.int64)

    # Match each board frame by first_sample // hop
    expected_n = len(ref)
    seen = set()
    dup_frames = 0
    unexpected_frames = 0
    offset_mismatches = 0
    diffs = []
    for i, (fn, fs_off) in enumerate(zip(fnums, fsamp)):
        if fs_off != fn * hop:
            offset_mismatches += 1
        if fn in seen:
            dup_frames += 1
        seen.add(int(fn))
        if 0 <= fn < expected_n:
            diffs.append(np.abs(got_feat[i] - ref[fn]))
        else:
            unexpected_frames += 1

    missing_frames = expected_n - len(seen & set(range(expected_n)))
    nonfinite_count = int(np.sum(~np.isfinite(got_feat)))
    diff = np.vstack(diffs) if diffs else np.array([[np.nan]])

    row = {
        "label": label,
        "run": run_idx,
        "port": port,
        "clips_sec": clips_sec,
        "win": win,
        "hop": hop,
        "mels": mels,
        "mfcc": mfcc,
        "speed": speed,
        "expected_features": r["expected_features"],
        "received_features": len(got_feat),
        "lost_features": r["lost_features"],
        "missing_frames": missing_frames,
        "dup_frames": dup_frames,
        "unexpected_frames": unexpected_frames,
        "offset_mismatches": offset_mismatches,
        "nonfinite_count": nonfinite_count,
        "bad_crc": brd.get("bad_crc", 0),
        "seq_gaps": brd.get("seq_gaps", 0),
        "uart_errors": brd.get("uart_errors", 0),
        "samples_received": brd.get("samples", 0),
        "compute_mean_us": round(float(np.mean(cu)), 1) if len(cu) else 0.0,
        "compute_median_us": round(float(np.median(cu)), 1) if len(cu) else 0.0,
        "compute_p95_us": round(float(np.percentile(cu, 95)), 1) if len(cu) else 0.0,
        "compute_max_us": round(float(np.max(cu)), 1) if len(cu) else 0.0,
        "lag_max_s": round(float(np.max(lag)), 4) if len(lag) else 0.0,
        "max_abs_err": round(float(np.max(diff)), 6),
        "mean_abs_err": round(float(np.mean(diff)), 6),
        "rmse": round(float(np.sqrt(np.mean(diff ** 2))), 6),
    }
    return row, ref, got_feat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["baseline", "optimized", "validate"], required=True)
    args = ap.parse_args()

    if args.mode == "baseline":
        print("=== Baseline Benchmark (starter features.h) ===")
        rows = []
        for port, baud in [(COM_PORT, 921600), (USB_PORT, 115200)]:
            row, _, _ = run_benchmark(port, baud, 2, 480, 320, 512, 40, 0, 1.0, "baseline_2s")
            rows.append(row)
            print(row)
        # Slow-paced stream to isolate numerical accuracy if speed=1.0 dropped frames
        row_slow, ref, got = run_benchmark(COM_PORT, 921600, 2, 480, 320, 512, 40, 0, 0.1, "baseline_2s_slow")
        rows.append(row_slow)
        print(row_slow)
        save_csv(OUT_DIR / "baseline_perf.csv", rows)

    elif args.mode == "optimized":
        print("=== Optimized Benchmark (Radix-2 FFT features.h) ===")
        rows = []
        ref_save, got_save = None, None
        for run_idx in range(1, 4):
            for port, baud in [(COM_PORT, 921600), (USB_PORT, 115200)]:
                row, ref, got = run_benchmark(port, baud, 10, 480, 320, 512, 40, 0, 1.0, "opt_logmel_10s", run_idx)
                rows.append(row)
                print(row)
                if port == COM_PORT and run_idx == 1:
                    ref_save, got_save = ref, got
        # Win 25ms / hop 10ms / 13 MFCCs
        row_mfcc, _, _ = run_benchmark(COM_PORT, 921600, 10, 400, 160, 512, 40, 13, 1.0, "opt_mfcc13_10s", 1)
        rows.append(row_mfcc)
        print(row_mfcc)
        save_csv(OUT_DIR / "optimized_perf.csv", rows)

        if ref_save is not None and got_save is not None:
            n_cmp = min(len(ref_save), len(got_save))
            err = np.abs(got_save[:n_cmp] - ref_save[:n_cmp])
            fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
            axes[0].imshow(ref_save[:n_cmp].T, aspect="auto", origin="lower")
            axes[0].set_title("NumPy Reference Log-Mel (dB)")
            axes[0].set_ylabel("Mel Band")
            axes[1].imshow(got_save[:n_cmp].T, aspect="auto", origin="lower")
            axes[1].set_title("ESP32-S3 Optimized Log-Mel (dB)")
            axes[1].set_ylabel("Mel Band")
            im = axes[2].imshow(err.T, aspect="auto", origin="lower", cmap="hot")
            axes[2].set_title(f"Absolute Error (max={np.max(err):.4f} dB, mean={np.mean(err):.5f} dB)")
            axes[2].set_xlabel("Frame")
            axes[2].set_ylabel("Mel Band")
            fig.colorbar(im, ax=axes[2], orientation="vertical", fraction=0.02)
            fig.tight_layout()
            fig.savefig(OUT_DIR / "feature_heatmaps_and_error.png", dpi=150)
            plt.close(fig)

    elif args.mode == "validate":
        from team_audio import CLASSES, clip_list
        print("=== 1. Six-Class Numerical Accuracy (1s per class, speed=1.0) ===")
        cls_rows = []
        for c_idx, cname in enumerate(CLASSES):
            cid = c_idx * 24
            x = get_clip(TEAM, cid, fs=16000, antialias=True)
            _, xhat, _ = quantise(x, fmt="i16", scale=1.0)
            pl = pack(x, fmt="i16", scale=1.0)
            ref = features(xhat, fs=16000, win=480, hop=320, nfft=512, mels=40, mfcc=0)
            b = Board(COM_PORT, baud=921600)
            try:
                r = b.stream(pl, fs=16000, fmt="i16", scale=1.0, win=480, hop=320,
                             nfft=512, mels=40, mfcc=0, feat=True, store="none",
                             speed=1.0, samples_per_packet=320)
            finally:
                b.close()
            got = r["features"]
            fnums = np.asarray(r["frame_numbers"], dtype=np.int64)
            err = np.abs(got - ref[fnums])
            cls_rows.append({
                "class": cname,
                "clip_id": cid,
                "frames": len(got),
                "max_abs_err_db": round(float(np.max(err)), 6),
                "mean_abs_err_db": round(float(np.mean(err)), 6),
                "rmse_db": round(float(np.sqrt(np.mean(err ** 2))), 6),
                "within_0_05db": bool(np.max(err) <= 0.05),
            })
            print(cls_rows[-1])
        save_csv(OUT_DIR / "six_class_accuracy.csv", cls_rows)

        print("\n=== 2. MFCC=13 Benchmark (3 runs, 10s, win=400, hop=160) ===")
        mfcc_rows = []
        for run_idx in range(1, 4):
            row, _, _ = run_benchmark(COM_PORT, 921600, 10, 400, 160, 512, 40, 13, 1.0, "opt_mfcc13_10s", run_idx)
            mfcc_rows.append(row)
            print(row)
        save_csv(OUT_DIR / "mfcc13_3runs.csv", mfcc_rows)

        print("\n=== 3. Full 144-Clip Real-Time Log-Mel Stream (COM5, speed=1.0) ===")
        clips = clip_list(TEAM)
        x_all = np.concatenate([get_clip(TEAM, cid, fs=16000, antialias=True) for cid, _ in clips])
        _, xhat_all, _ = quantise(x_all, fmt="i16", scale=1.0)
        pl_all = pack(x_all, fmt="i16", scale=1.0)
        ref_all = features(xhat_all, fs=16000, win=480, hop=320, nfft=512, mels=40, mfcc=0)
        b = Board(COM_PORT, baud=921600)
        try:
            r_all = b.stream(pl_all, fs=16000, fmt="i16", scale=1.0, win=480, hop=320,
                             nfft=512, mels=40, mfcc=0, feat=True, store="none",
                             speed=1.0, samples_per_packet=320)
        finally:
            b.close()
        brd_all = r_all["board"] or {}
        fsamp_all = np.asarray(r_all["first_sample"], dtype=np.int64)
        boundary_frames = int(np.sum((fsamp_all // 16000) != ((fsamp_all + 480 - 1) // 16000)))
        err_all = np.abs(r_all["features"] - ref_all[np.asarray(r_all["frame_numbers"], dtype=np.int64)])
        full_summary = {
            "clips": len(clips),
            "samples_expected": len(x_all),
            "samples_received": brd_all.get("samples", 0),
            "expected_features": r_all["expected_features"],
            "received_features": len(r_all["features"]),
            "lost_features": r_all["lost_features"],
            "boundary_crossing_frames": boundary_frames,
            "bad_crc": brd_all.get("bad_crc", 0),
            "seq_gaps": brd_all.get("seq_gaps", 0),
            "uart_errors": brd_all.get("uart_errors", 0),
            "lag_max_s": round(float(np.max(r_all["lag_s"])), 4),
            "compute_median_us": round(float(np.median(r_all["compute_us"])), 1),
            "max_abs_err_db": round(float(np.max(err_all)), 6),
        }
        print(full_summary)
        save_csv(OUT_DIR / "full_144_clips_realtime.csv", [full_summary])

        # Baseline vs Optimized Computation Time Graph
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        labels = ["Baseline (DFT)", "Optimized (Radix-2 FFT)", "Optimized (MFCC-13)"]
        vals = [1855.7, 1.711, float(np.median([r["compute_median_us"] for r in mfcc_rows])) / 1000.0]
        bars = ax.bar(labels, vals, color=["tab:red", "tab:green", "tab:blue"])
        ax.set_yscale("log")
        ax.axhline(20.0, color="black", linestyle="--", label="20 ms Hop Budget (Log-Mel)")
        ax.set_ylabel("Median Compute Time per Frame (ms, log scale)")
        ax.set_title("Mission 3: Baseline DFT vs Optimized Radix-2 FFT")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v * 1.15, f"{v:.3f} ms", ha="center", fontsize=9)
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT_DIR / "baseline_vs_optimized.png", dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    main()
