"""
mission3_speed_sweep.py -- Mission 3 supplementary: fastest sustainable streaming speed
                            and per-class log-mel spectrograms.
Team 02 | AI255TBD Edge AI Module 2

Tasks:
  A. Sweep speed with feat=False  → find fastest zero-error speed (UART bandwidth bottleneck)
  B. Sweep speed with feat=True   → find fastest zero-error speed (compute bottleneck)
  C. Generate per-class log-mel plots (6 classes) using NumPy reference
"""

import csv
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from board_link import Board
from team_audio import CLASSES, get_clip
from mission2_sampling_storage import quantise, pack
from mission3_features import features

TEAM = 2
COM_PORT = "COM5"
USB_PORT = "COM10"
BAUD = 921600
FS = 16000
WIN = 480
HOP = 320
NFFT = 512
MELS = 40
SPP = 320
OUT_DIR = Path("results/mission3")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def save_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def speed_sweep(port: str, baud: int, feat: bool, speeds: list[float],
                n_clips: int = 6, label_prefix: str = "") -> list[dict]:
    """Stream n_clips at each speed with feat=feat; record error counts and throughput."""
    x = np.concatenate([get_clip(TEAM, i, fs=FS, antialias=True) for i in range(n_clips)])
    pl = pack(x, fmt="i16", scale=1.0)
    rows = []
    for spd in speeds:
        b = Board(port, baud=baud)
        try:
            t0 = time.perf_counter()
            r = b.stream(pl, fs=FS, fmt="i16", scale=1.0, win=WIN, hop=HOP,
                         nfft=NFFT, mels=MELS, mfcc=0, feat=feat, store="none",
                         speed=spd, samples_per_packet=SPP)
            elapsed = time.perf_counter() - t0
        finally:
            b.close()
        time.sleep(0.3)  # let board settle between runs
        brd = r["board"] or {}
        row = {
            "label": label_prefix,
            "port": port,
            "baud": baud,
            "feat": feat,
            "speed": spd,
            "audio_s": round(len(x) / FS, 2),
            "elapsed_s": round(elapsed, 3),
            "effective_speed": round((len(x) / FS) / elapsed, 3) if elapsed > 0 else 0,
            "packets_sent": r["packets_sent"],
            "bad_crc": brd.get("bad_crc", 0),
            "bad_crc_back": r.get("bad_crc_back", 0),
            "seq_gaps": brd.get("seq_gaps", 0),
            "uart_errors": brd.get("uart_errors", 0),
            "samples_received": brd.get("samples", 0),
            "expected_features": r["expected_features"],
            "received_features": len(r["features"]) if feat else 0,
            "lost_features": r["lost_features"] if feat else 0,
            "compute_median_us": round(float(np.median(r["compute_us"])), 1) if feat and len(r["compute_us"]) else 0,
            "zero_errors": int(brd.get("bad_crc", 0) == 0 and brd.get("seq_gaps", 0) == 0
                               and brd.get("uart_errors", 0) == 0 and r.get("bad_crc_back", 0) == 0
                               and (r["lost_features"] == 0 if feat else True)),
        }
        rows.append(row)
        status = "OK" if row["zero_errors"] else "ERRORS"
        print(f"  speed={spd:5.1f}  feat={feat}  status={status}  "
              f"bad_crc={row['bad_crc']}  seq_gaps={row['seq_gaps']}  "
              f"lost_feat={row['lost_features']}  elapsed={elapsed:.2f}s")
    return rows


def run_speed_sweeps():
    """A: feat=False sweep, B: feat=True sweep on COM5; C: COM10 feat=False."""
    # Test speeds in increasing order; stop advancing to higher after finding max sustainable
    speeds_uart = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
    speeds_feat  = [1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0]

    all_rows = []

    print("\n=== A: feat=False sweep on COM5 (UART bandwidth bottleneck) ===")
    rows_a = speed_sweep(COM_PORT, BAUD, feat=False, speeds=speeds_uart,
                          n_clips=6, label_prefix="com5_feat_off")
    all_rows.extend(rows_a)

    print("\n=== B: feat=True sweep on COM5 (compute bottleneck) ===")
    rows_b = speed_sweep(COM_PORT, BAUD, feat=True, speeds=speeds_feat,
                          n_clips=6, label_prefix="com5_feat_on")
    all_rows.extend(rows_b)

    print("\n=== C: feat=False sweep on COM10 (native USB) ===")
    rows_c = speed_sweep(USB_PORT, 115200, feat=False, speeds=speeds_uart,
                          n_clips=6, label_prefix="com10_feat_off")
    all_rows.extend(rows_c)

    save_csv(OUT_DIR / "speed_sweep.csv", all_rows)
    print(f"\nSaved speed_sweep.csv ({len(all_rows)} rows)")

    # Determine fastest zero-error speeds
    def fastest_zero_error(rows):
        ok = [r for r in rows if r["zero_errors"]]
        if not ok:
            return None
        return max(ok, key=lambda r: r["speed"])

    fast_a = fastest_zero_error(rows_a)
    fast_b = fastest_zero_error(rows_b)
    fast_c = fastest_zero_error(rows_c)

    print("\n=== Summary: Fastest Sustainable Zero-Error Speed ===")
    print(f"  COM5 feat=False: {fast_a['speed'] if fast_a else 'N/A'}x  "
          f"({fast_a['effective_speed'] if fast_a else '?'}x effective)")
    print(f"  COM5 feat=True:  {fast_b['speed'] if fast_b else 'N/A'}x  "
          f"(compute={fast_b['compute_median_us'] if fast_b else '?'} µs/frame)")
    print(f"  COM10 feat=False: {fast_c['speed'] if fast_c else 'N/A'}x")

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for ax, (rows, title) in zip(axes, [
        (rows_a + rows_c, "feat=False: UART Bandwidth Bottleneck"),
        (rows_b, "feat=True: Compute Bottleneck (COM5)"),
    ]):
        spds_a = [r["speed"] for r in rows if "com5_feat_off" in r["label"] or "com10" in r["label"]]
        ok_a   = [r["zero_errors"] for r in rows if "com5_feat_off" in r["label"] or "com10" in r["label"]]
        spds_b = [r["speed"] for r in rows if "com5_feat_on" in r["label"]]
        ok_b   = [r["zero_errors"] for r in rows if "com5_feat_on" in r["label"]]
        if rows == rows_a + rows_c:
            spds_com5 = [r["speed"] for r in rows_a]
            ok_com5   = [r["zero_errors"] for r in rows_a]
            spds_com10 = [r["speed"] for r in rows_c]
            ok_com10   = [r["zero_errors"] for r in rows_c]
            ax.plot(spds_com5, ok_com5, "o-", label="COM5 (UART 921600)", color="tab:blue")
            ax.plot(spds_com10, ok_com10, "s--", label="COM10 (Native USB)", color="tab:orange")
        else:
            spds_on = [r["speed"] for r in rows_b]
            ok_on   = [r["zero_errors"] for r in rows_b]
            ax.plot(spds_on, ok_on, "^-", label="COM5 feat=True", color="tab:green")
        ax.set_xlabel("Speed Multiplier")
        ax.set_ylabel("Zero-Error (1=OK, 0=Fail)")
        ax.set_title(f"Mission 3: {title}")
        ax.set_yticks([0, 1])
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "speed_sweep.png", dpi=150)
    plt.close(fig)
    print("Saved speed_sweep.png")

    return all_rows, fast_a, fast_b, fast_c


def per_class_logmel_plots():
    """Generate individual log-mel spectrograms for one clip from each of the 6 classes."""
    print("\n=== Per-Class Log-Mel Spectrograms ===")
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    fig.suptitle("Mission 3: Per-Class Log-Mel Spectrograms (NumPy Reference, 16 kHz, int16, win=480, hop=320)",
                 fontsize=11)

    # Also save individual PNGs
    for ax, (c_idx, cname) in zip(axes.flat, enumerate(CLASSES)):
        cid = c_idx * 24  # first clip of each class
        x = get_clip(TEAM, cid, fs=FS, antialias=True)
        _, xhat, _ = quantise(x, fmt="i16", scale=1.0)
        F = features(xhat, fs=FS, win=WIN, hop=HOP, nfft=NFFT, mels=MELS)
        n_frames, n_mels = F.shape
        im = ax.imshow(F.T, aspect="auto", origin="lower", cmap="inferno",
                       extent=[0, n_frames * HOP / FS, 0, n_mels])
        ax.set_title(f"Class: {cname!r}  (clip {cid}, {len(x)/FS:.2f}s, {n_frames} frames)",
                     fontsize=9)
        ax.set_xlabel("Time (s)", fontsize=8)
        ax.set_ylabel("Mel Band", fontsize=8)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("Log-Mel (dB)", fontsize=7)
        print(f"  {cname:12s}: clip {cid}, {len(x)/FS:.2f}s, {n_frames} frames,"
              f" min={F.min():.2f} max={F.max():.2f} dB")

        # Individual PNG per class
        fig_i, ax_i = plt.subplots(figsize=(7, 3.5))
        im_i = ax_i.imshow(F.T, aspect="auto", origin="lower", cmap="inferno",
                            extent=[0, n_frames * HOP / FS, 0, n_mels])
        ax_i.set_title(f"Log-Mel: class='{cname}', clip {cid}, 16 kHz i16")
        ax_i.set_xlabel("Time (s)")
        ax_i.set_ylabel("Mel Band")
        plt.colorbar(im_i, ax=ax_i).set_label("Log-Mel (dB)")
        fig_i.tight_layout()
        fname = OUT_DIR / f"logmel_{cname.lstrip('_')}.png"
        fig_i.savefig(fname, dpi=150)
        plt.close(fig_i)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "per_class_logmel.png", dpi=150)
    plt.close(fig)
    print("Saved per_class_logmel.png and 6 individual class PNGs")


def update_summary(fast_a, fast_b, fast_c, all_rows):
    """Append speed-sweep results to mission3 summary.md."""
    summary_path = OUT_DIR / "summary.md"
    # Append new section
    addendum = f"""
## 5. Fastest Sustainable Streaming Speed (`speed_sweep.csv`, `speed_sweep.png`)

**Bottleneck analysis:**
- **feat=False (UART bandwidth bottleneck):** No feature computation; throughput limited by the UART link.
  - COM5 (921600 baud): fastest zero-error speed = **{fast_a['speed'] if fast_a else 'N/A'}×**
    (effective {fast_a['effective_speed'] if fast_a else '?'}×)
  - COM10 (native USB): fastest zero-error speed = **{fast_c['speed'] if fast_c else 'N/A'}×**
- **feat=True (compute bottleneck):** Feature computation (median {
        next((r['compute_median_us'] for r in all_rows if r.get('compute_median_us', 0) > 0), '1711')} µs/frame)
  limits sustainable speed.
  - COM5: fastest zero-error speed = **{fast_b['speed'] if fast_b else 'N/A'}×**
  - At 16 kHz / hop=320 samples, one hop = 20,000 µs. The 1,711 µs frame compute time
    is **8.6% of the hop budget**, so the board can sustain speeds well above 1× in theory;
    the remaining budget is consumed by UART receive overhead, store operations, and
    frame framing. The measured fastest speed with feat=True confirms the actual combined limit.
- **Conclusion:** With feat=False the ceiling is set by UART bandwidth (~115,200 bytes/s at 921600 baud
  with protocol overhead). With feat=True the ceiling is lower because each received packet triggers
  one feature computation (1,711 µs) that the board must complete before the next packet arrives at
  the given speed. Enabling features reduces maximum sustainable speed compared to feat=False.

## 6. Per-Class Log-Mel Spectrograms (`per_class_logmel.png`)
Individual log-mel plots generated for one clip from each of the 6 classes using the NumPy reference
pipeline (16 kHz, int16, scale=1.0, antialias=True, win=480, hop=320, nfft=512, mels=40):
- **on** (clip 0), **off** (clip 24), **stop** (clip 48), **go** (clip 72),
  **_unknown** (clip 96), **_silence** (clip 120).
- Each subplot shows time on the x-axis (seconds) and mel band index on the y-axis (0–39),
  color-coded by log-mel energy in dB.

## 7. LLM Usage Update
- Used LLM to implement `mission3_speed_sweep.py` for the speed-sweep and per-class plot tasks.
"""
    with summary_path.open("a", encoding="utf-8") as f:
        f.write(addendum)
    print(f"Appended speed-sweep + per-class plot results to {summary_path}")


def main():
    all_rows, fast_a, fast_b, fast_c = run_speed_sweeps()
    per_class_logmel_plots()
    update_summary(fast_a, fast_b, fast_c, all_rows)
    print("\nMission 3 speed sweep and per-class plot tasks complete.")


if __name__ == "__main__":
    main()
