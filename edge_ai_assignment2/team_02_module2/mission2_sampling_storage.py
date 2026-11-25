"""
mission2_sampling_storage.py -- Mission 2: Sampling, Quantisation, and Storage (Tasks 1-4)
Team 02 | AI255TBD Edge AI Module 2
"""

import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from board_link import Board
from team_audio import CLASSES, get_clip

TEAM = 2
PORT = "COM5"
BAUD = 921600
OUT_DIR = Path("results/mission2")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------ Task 1: Quantisation
def quantise(x: np.ndarray, fmt: str = "i16", scale: float = 1.0) -> tuple[np.ndarray, np.ndarray, int]:
    """Quantise float32 audio `x` into `fmt` ('f32', 'i16', 'i8') with scale `S`.
    Returns (q_array, xhat_float32, clipped_count).
    """
    x = np.asarray(x, dtype=np.float32)
    if fmt == "f32":
        return x.astype("<f4"), x.astype(np.float32), 0
    bits = 16 if fmt == "i16" else 8
    qmax = (1 << (bits - 1)) - 1
    qmin = -(1 << (bits - 1))
    scaled = np.round(x / scale * qmax)
    clipped = int(np.sum((scaled < qmin) | (scaled > qmax)))
    q = np.clip(scaled, qmin, qmax).astype("<i2" if fmt == "i16" else np.int8)
    xhat = (q.astype(np.float32) * scale / qmax).astype(np.float32)
    return q, xhat, clipped


def pack(x: np.ndarray, fmt: str = "i16", scale: float = 1.0) -> bytes:
    """Quantise and pack `x` into little-endian wire bytes."""
    q, _, _ = quantise(x, fmt=fmt, scale=scale)
    return q.tobytes()


def unpack(buf: bytes, fmt: str = "i16", scale: float = 1.0) -> np.ndarray:
    """Unpack wire bytes back into float32 audio."""
    if fmt == "f32":
        return np.frombuffer(buf, dtype="<f4").astype(np.float32)
    if fmt == "i16":
        return (np.frombuffer(buf, dtype="<i2").astype(np.float32) * scale / 32767.0).astype(np.float32)
    if fmt == "i8":
        return (np.frombuffer(buf, dtype=np.int8).astype(np.float32) * scale / 127.0).astype(np.float32)
    raise ValueError(f"Unsupported fmt: {fmt}")


# ------------------------------------------------------------------ Task 2: Sampling & Aliasing
def run_aliasing_experiment() -> dict:
    silence_id = CLASSES.index("_silence") * 24  # clip 120
    x48 = get_clip(TEAM, silence_id, fs=48000, antialias=False)
    x8_raw = get_clip(TEAM, silence_id, fs=8000, antialias=False)
    x8_aa = get_clip(TEAM, silence_id, fs=8000, antialias=True)

    def spec_db(x, fs):
        freqs = np.fft.rfftfreq(len(x), d=1.0 / fs)
        mag = np.abs(np.fft.rfft(x)) / len(x)
        db = 20.0 * np.log10(np.maximum(mag, 1e-12))
        return freqs, db

    f48, db48 = spec_db(x48, 48000)
    f8_raw, db8_raw = spec_db(x8_raw, 8000)
    f8_aa, db8_aa = spec_db(x8_aa, 8000)

    # Find regulator whine peak above 1 kHz in 48 kHz spectrum
    mask48 = f48 >= 1000.0
    f_whine = float(f48[mask48][np.argmax(db48[mask48])])
    f_alias_pred = abs(f_whine - 8000.0 * round(f_whine / 8000.0))
    mask8 = f8_raw >= 500.0
    f_alias_meas = float(f8_raw[mask8][np.argmax(db8_raw[mask8])])

    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharey=True)
    axes[0].plot(f48 / 1000.0, db48, lw=0.8)
    axes[0].set_title(f"48 kHz Original Silence Spectrum (peak at {f_whine:.1f} Hz)")
    axes[0].set_xlabel("Frequency (kHz)")
    axes[0].set_ylabel("Magnitude (dB)")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(f8_raw / 1000.0, db8_raw, lw=0.8, color="tab:red")
    axes[1].set_title(f"8 kHz Without Anti-Alias Filter (aliased peak at {f_alias_meas:.1f} Hz; predicted {f_alias_pred:.1f} Hz)")
    axes[1].set_xlabel("Frequency (kHz)")
    axes[1].set_ylabel("Magnitude (dB)")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(f8_aa / 1000.0, db8_aa, lw=0.8, color="tab:green")
    axes[2].set_title("8 kHz With Anti-Alias Filter (antialias=True)")
    axes[2].set_xlabel("Frequency (kHz)")
    axes[2].set_ylabel("Magnitude (dB)")
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "aliasing_spectra.png", dpi=150)
    plt.close(fig)

    res = {
        "silence_clip_id": silence_id,
        "whine_freq_hz": f_whine,
        "aliased_pred_hz": f_alias_pred,
        "aliased_meas_hz": f_alias_meas,
    }
    print("=== Task 2: Aliasing ===", res)
    return res


# ------------------------------------------------------------------ Task 3: SQNR
def run_sqnr_experiment() -> list[dict]:
    word_id = 0  # "on"
    silence_id = CLASSES.index("_silence") * 24  # 120
    clips = [("word_on", word_id), ("silence", silence_id)]
    configs = [("i16", 1.0), ("i8", 1.0), ("i8", 0.25), ("i8", 0.1)]
    rows = []
    for cname, cid in clips:
        x = get_clip(TEAM, cid, fs=16000, antialias=True)
        sig_pow = float(np.mean(x.astype(np.float64) ** 2))
        for fmt, scale in configs:
            _, xhat, clipped = quantise(x, fmt=fmt, scale=scale)
            err = x.astype(np.float64) - xhat.astype(np.float64)
            err_pow = float(np.mean(err ** 2))
            sqnr = 10.0 * np.log10(sig_pow / err_pow) if err_pow > 0 else float("inf")
            rmse = float(np.sqrt(err_pow))
            bps = 2 if fmt == "i16" else 1
            rows.append({
                "clip": cname,
                "clip_id": cid,
                "fmt": fmt,
                "scale": scale,
                "sqnr_db": round(sqnr, 2),
                "rmse": round(rmse, 6),
                "clipped_samples": clipped,
                "bytes_144_clips_16khz": 144 * 16000 * bps,
            })
    with (OUT_DIR / "sqnr_measurements.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    labels = [f"{r['fmt']} (S={r['scale']})" for r in rows[:4]]
    x_idx = np.arange(len(labels))
    w_bar = 0.35
    ax.bar(x_idx - w_bar / 2, [r["sqnr_db"] for r in rows[:4]], w_bar, label="Word ('on')")
    ax.bar(x_idx + w_bar / 2, [r["sqnr_db"] for r in rows[4:]], w_bar, label="Silence")
    ax.set_xticks(x_idx)
    ax.set_xticklabels(labels)
    ax.set_ylabel("SQNR (dB)")
    ax.set_title("Mission 2: SQNR by Format and Scale (16 kHz)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "sqnr_comparison.png", dpi=150)
    plt.close(fig)

    print("=== Task 3: SQNR ===")
    for r in rows:
        print(r)
    return rows


# ------------------------------------------------------------------ Task 4: SRAM & PSRAM Capacity
def run_memory_capacity_experiment() -> list[dict]:
    b = Board(PORT, baud=BAUD)
    rows = []
    try:
        info = b.info()
        print("=== Task 4: Board INFO ===", info)
        # Test SRAM and PSRAM with 16 kHz i16 (32,000 B/s)
        for where, test_kb_list in [
            ("sram", [64, 128, 192, 224, 240]),
            ("psram", [1024, 4096, 7680, 8000, 8192]),
        ]:
            for kb in test_kb_list:
                n_bytes = kb * 1024
                n_samples = n_bytes // 2
                # Generate deterministic dummy i16 payload of exact length
                pl = np.zeros(n_samples, dtype="<i2").tobytes()
                b.free()
                try:
                    r = b.stream(pl, fs=16000, fmt="i16", feat=False, store=where,
                                 max_bytes=n_bytes, speed=0.0, samples_per_packet=1024)
                    brd = r["board"]
                    ok = (brd.get("stored", 0) == n_bytes and brd.get("store_full", 0) == 0)
                    rows.append({
                        "store": where,
                        "requested_bytes": n_bytes,
                        "requested_kb": kb,
                        "stored_bytes": brd.get("stored", 0),
                        "samples_received": brd.get("samples", 0),
                        "store_full": brd.get("store_full", 0),
                        "success": ok,
                        "error": "",
                    })
                except Exception as e:
                    rows.append({
                        "store": where,
                        "requested_bytes": n_bytes,
                        "requested_kb": kb,
                        "stored_bytes": 0,
                        "samples_received": 0,
                        "store_full": 0,
                        "success": False,
                        "error": str(e),
                    })
                print(rows[-1])
        b.free()
    finally:
        b.close()

    with (OUT_DIR / "memory_capacity.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return rows


def main():
    run_aliasing_experiment()
    run_sqnr_experiment()
    run_memory_capacity_experiment()


if __name__ == "__main__":
    main()
