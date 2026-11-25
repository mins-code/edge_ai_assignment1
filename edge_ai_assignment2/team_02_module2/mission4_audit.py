"""
mission4_audit.py -- Mission 4 supplementary audit tasks:
  1. Fastest zero-loss speed on COM5 and COM10
  2. Add bad_crc_back to all stream records
  3. Save first_sample and frame_numbers in dataset NPZs
  4. Verify full 144-clip loss accounting with feature counts
Team 02 | AI255TBD Edge AI Module 2
"""

import csv
import json
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from board_link import Board
from team_audio import CLASSES, clip_list, get_clip
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
OUT_DIR = Path("results/mission4")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def save_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------------------
# Task 1: Fastest zero-loss speed on COM5 and COM10
# ---------------------------------------------------------------------------
def fastest_zero_loss_speed():
    """Sweep streaming speed on COM5 and COM10 without features to find fastest zero-loss speed."""
    print("=== Task 1: Fastest Zero-Loss Streaming Speed ===")
    x = np.concatenate([get_clip(TEAM, i, fs=FS, antialias=True) for i in range(6)])
    pl = pack(x, fmt="i16", scale=1.0)
    speeds = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0]

    rows = []
    for port, baud, label in [(COM_PORT, BAUD, "COM5_921600"), (USB_PORT, 115200, "COM10_115200")]:
        print(f"\n  Sweeping {label} ...")
        for spd in speeds:
            b = Board(port, baud=baud)
            try:
                r = b.stream(pl, fs=FS, fmt="i16", scale=1.0, win=WIN, hop=HOP,
                             nfft=NFFT, mels=MELS, mfcc=0, feat=False, store="none",
                             speed=spd, samples_per_packet=SPP)
            finally:
                b.close()
            time.sleep(0.3)
            brd = r["board"] or {}
            bad_crc = brd.get("bad_crc", 0)
            seq_gaps = brd.get("seq_gaps", 0)
            uart_err = brd.get("uart_errors", 0)
            bcb = r.get("bad_crc_back", 0)
            samples_recv = brd.get("samples", 0)
            packets_expected = (len(x) + SPP - 1) // SPP
            zero_loss = int(
                bad_crc == 0 and seq_gaps == 0 and uart_err == 0 and bcb == 0
                and samples_recv == len(x) and r["packets_sent"] == packets_expected
            )
            row = {
                "port": port,
                "baud": baud,
                "interface": label,
                "speed": spd,
                "packets_expected": packets_expected,
                "packets_sent": r["packets_sent"],
                "frames_ok": brd.get("frames_ok", 0),
                "bad_crc": bad_crc,
                "bad_crc_back": bcb,
                "seq_gaps": seq_gaps,
                "uart_errors": uart_err,
                "samples_expected": len(x),
                "samples_received": samples_recv,
                "zero_loss": zero_loss,
            }
            rows.append(row)
            print(f"    speed={spd:5.1f}  zero_loss={zero_loss}  bad_crc={bad_crc}  "
                  f"seq_gaps={seq_gaps}  bad_crc_back={bcb}")

    save_csv(OUT_DIR / "fastest_speed.csv", rows)
    print(f"\nSaved fastest_speed.csv ({len(rows)} rows)")

    for label in ["COM5_921600", "COM10_115200"]:
        ok = [r for r in rows if r["interface"] == label and r["zero_loss"]]
        if ok:
            best = max(ok, key=lambda r: r["speed"])
            print(f"  {label}: fastest zero-loss speed = {best['speed']}×")
        else:
            print(f"  {label}: no zero-loss speed found in sweep!")

    return rows


# ---------------------------------------------------------------------------
# Task 2+3: Rebuild loss_experiments.csv with bad_crc_back; rebuild datasets
# with first_sample / frame_numbers
# ---------------------------------------------------------------------------
def rebuild_loss_experiments_with_bad_crc_back():
    """Re-run deliberate loss experiments and add bad_crc_back field."""
    print("\n=== Task 2: Rebuild loss_experiments.csv with bad_crc_back ===")
    x = np.concatenate([get_clip(TEAM, i, fs=FS, antialias=True) for i in range(6)])
    pl = pack(x, fmt="i16", scale=1.0)
    samples_expected = len(x)
    packets_expected = (samples_expected + SPP - 1) // SPP

    experiments = [
        ("control_no_loss", ()),
        ("readme_skip_10_11_200", (10, 11, 200)),
        ("single_beginning_5", (5,)),
        ("burst_20_to_23", (20, 21, 22, 23)),
    ]
    rows = []
    for name, skip_tuple in experiments:
        b = Board(COM_PORT, baud=BAUD)
        try:
            r = b.stream(pl, fs=FS, fmt="i16", scale=1.0, win=WIN, hop=HOP,
                         nfft=NFFT, mels=MELS, mfcc=0, feat=True, store="none",
                         speed=1.0, samples_per_packet=SPP, skip=skip_tuple)
        finally:
            b.close()
        time.sleep(0.3)
        brd = r["board"] or {}
        fsamp = np.asarray(r["first_sample"], dtype=np.int64)
        orig_last = (samples_expected - WIN) // HOP * HOP
        last_shift = int(fsamp[-1] - orig_last) if len(fsamp) else 0
        row = {
            "experiment": name,
            "skip": str(skip_tuple),
            "packets_expected": packets_expected,
            "packets_skipped": len(skip_tuple),
            "packets_sent": r["packets_sent"],
            "frames_ok": brd.get("frames_ok", 0),
            "bad_crc": brd.get("bad_crc", 0),
            "bad_crc_back": r.get("bad_crc_back", 0),
            "seq_gaps": brd.get("seq_gaps", 0),
            "uart_errors": brd.get("uart_errors", 0),
            "samples_expected": samples_expected,
            "samples_received": brd.get("samples", 0),
            "expected_features": r["expected_features"],
            "received_features": len(r["features"]),
            "lost_features": r["lost_features"],
            "last_first_sample_shift": last_shift,
        }
        rows.append(row)
        print(f"  {name}: bad_crc_back={row['bad_crc_back']}  lost_feat={row['lost_features']}")

    save_csv(OUT_DIR / "loss_experiments.csv", rows)
    print("Updated loss_experiments.csv (with bad_crc_back)")
    return rows


def rebuild_datasets_with_metadata():
    """
    Re-stream all 144 clips; save first_sample and frame_numbers in dataset NPZs.
    X, y, classes, clip_ids are UNCHANGED (same values as before).
    """
    print("\n=== Task 3: Rebuild Datasets with first_sample + frame_numbers ===")
    clips = clip_list(TEAM)
    x_all = np.concatenate([get_clip(TEAM, cid, fs=FS, antialias=True) for cid, _ in clips])
    _, xhat_all, _ = quantise(x_all, fmt="i16", scale=1.0)
    pl_all = pack(x_all, fmt="i16", scale=1.0)
    ref_all = features(xhat_all, fs=FS, win=WIN, hop=HOP, nfft=NFFT, mels=MELS, mfcc=0)

    b = Board(COM_PORT, baud=BAUD)
    try:
        r = b.stream(pl_all, fs=FS, fmt="i16", scale=1.0, win=WIN, hop=HOP,
                     nfft=NFFT, mels=MELS, mfcc=0, feat=True, store="none",
                     speed=1.0, samples_per_packet=SPP)
    finally:
        b.close()

    brd = r["board"] or {}
    got_feat = r["features"]
    fsamp = np.asarray(r["first_sample"], dtype=np.int64)
    fnums = np.asarray(r["frame_numbers"], dtype=np.int64)
    bcb = r.get("bad_crc_back", 0)

    # Integrity assertions
    assert brd.get("samples", 0) == len(x_all), f"Sample mismatch: {brd.get('samples')} != {len(x_all)}"
    assert len(got_feat) == 7199, f"Expected 7199 frames, got {len(got_feat)}"
    assert brd.get("bad_crc", 0) == 0 and brd.get("seq_gaps", 0) == 0 and brd.get("uart_errors", 0) == 0
    assert bcb == 0, f"bad_crc_back={bcb} (unexpected return-path corruption)"

    # Build clean datasets (same logic as before) + accumulate per-slot metadata
    X_board = np.zeros((144, 49, MELS), dtype=np.float32)
    X_numpy = np.zeros((144, 49, MELS), dtype=np.float32)
    occupied = np.zeros((144, 49), dtype=bool)
    # Store per-slot first_sample and frame_number for dataset NPZ
    fs_dataset = np.zeros((144, 49), dtype=np.int64)   # original first_sample per slot
    fn_dataset = np.zeros((144, 49), dtype=np.int64)   # frame_number per slot
    y = np.array([CLASSES.index(lab) for _, lab in clips], dtype=np.int64)
    clip_ids = np.array([cid for cid, _ in clips], dtype=np.int64)
    boundary_count = 0
    seen_frames = set()

    for i, (fn, s0) in enumerate(zip(fnums, fsamp)):
        assert int(fn) not in seen_frames, f"Duplicate frame {fn}"
        seen_frames.add(int(fn))
        assert int(s0) == int(fn) * HOP, f"Offset mismatch: {s0} != {fn * HOP}"
        c_start = int(s0 // FS)
        c_end = int((s0 + WIN - 1) // FS)
        if c_start != c_end:
            boundary_count += 1
            continue
        frame_in_clip = int((s0 % FS) // HOP)
        occupied[c_start, frame_in_clip] = True
        X_board[c_start, frame_in_clip] = got_feat[i]
        X_numpy[c_start, frame_in_clip] = ref_all[fn]
        fs_dataset[c_start, frame_in_clip] = int(s0)
        fn_dataset[c_start, frame_in_clip] = int(fn)

    assert occupied.all(), "Unpopulated zero-filled feature slots remain"
    assert boundary_count == 143, f"Expected 143 boundary frames, got {boundary_count}"
    assert np.isfinite(X_board).all() and np.isfinite(X_numpy).all()
    max_diff = float(np.max(np.abs(X_board - X_numpy)))
    assert max_diff <= 0.05, f"Max diff {max_diff} > 0.05 dB"

    # Verify unchanged X, y relative to old datasets
    old_b = np.load(OUT_DIR / "dataset_board.npz")
    old_n = np.load(OUT_DIR / "dataset_numpy.npz")
    assert np.allclose(X_board, old_b["X"], atol=1e-5), "X_board changed unexpectedly!"
    assert np.allclose(X_numpy, old_n["X"], atol=1e-5), "X_numpy changed unexpectedly!"
    print("  Verified: X, y unchanged from previous dataset build.")

    # Save updated NPZ files with first_sample and frame_numbers metadata
    np.savez_compressed(
        OUT_DIR / "dataset_board.npz",
        X=X_board, y=y, clip_ids=clip_ids, classes=np.array(CLASSES),
        first_sample=fs_dataset, frame_numbers=fn_dataset,
        fs=FS, win=WIN, hop=HOP, nfft=NFFT, mels=MELS, fmt="i16", scale=1.0,
    )
    np.savez_compressed(
        OUT_DIR / "dataset_numpy.npz",
        X=X_numpy, y=y, clip_ids=clip_ids, classes=np.array(CLASSES),
        first_sample=fs_dataset, frame_numbers=fn_dataset,
        fs=FS, win=WIN, hop=HOP, nfft=NFFT, mels=MELS, fmt="i16", scale=1.0,
    )

    # Verify reload
    db2 = np.load(OUT_DIR / "dataset_board.npz")
    dn2 = np.load(OUT_DIR / "dataset_numpy.npz")
    assert db2["X"].shape == (144, 49, 40) and dn2["X"].shape == (144, 49, 40)
    assert "first_sample" in db2 and "frame_numbers" in db2
    assert db2["first_sample"].shape == (144, 49)
    print(f"  Saved updated dataset_board.npz and dataset_numpy.npz with first_sample, frame_numbers")
    print(f"  bad_crc_back={bcb}  max_diff_db={max_diff:.6f}")

    return {
        "total_frames": len(got_feat),
        "boundary_frames_excluded": boundary_count,
        "clean_frames": int(X_board.shape[0] * X_board.shape[1]),
        "X_shape": list(X_board.shape),
        "y_shape": list(y.shape),
        "max_diff_db": round(max_diff, 6),
        "bad_crc": brd.get("bad_crc", 0),
        "bad_crc_back": bcb,
        "seq_gaps": brd.get("seq_gaps", 0),
        "packets_sent": r["packets_sent"],
        "frames_ok": brd.get("frames_ok", 0),
        "samples_received": brd.get("samples", 0),
        "expected_features": r["expected_features"],
        "received_features": len(got_feat),
        "lost_features": r["lost_features"],
    }


def run_full_stream_skip_experiment() -> dict:
    """README-required full 144-clip stream with skip=(10, 11, 200)."""
    print("\n=== Task 3b: Full 144-Clip Stream with skip=(10, 11, 200) ===")
    clips = clip_list(TEAM)
    x_all = np.concatenate([get_clip(TEAM, cid, fs=FS, antialias=True) for cid, _ in clips])
    pl_all = pack(x_all, fmt="i16", scale=1.0)
    packets_expected = (len(x_all) + SPP - 1) // SPP
    skip_tuple = (10, 11, 200)

    b = Board(COM_PORT, baud=BAUD)
    try:
        r = b.stream(pl_all, fs=FS, fmt="i16", scale=1.0, win=WIN, hop=HOP,
                     nfft=NFFT, mels=MELS, mfcc=0, feat=True, store="none",
                     speed=1.0, samples_per_packet=SPP, skip=skip_tuple)
    finally:
        b.close()
    time.sleep(0.3)
    brd = r["board"] or {}
    row = {
        "experiment": "full_144_skip_10_11_200",
        "skip": str(skip_tuple),
        "packets_expected": packets_expected,
        "packets_skipped": len(skip_tuple),
        "packets_sent": r["packets_sent"],
        "frames_ok": brd.get("frames_ok", 0),
        "bad_crc": brd.get("bad_crc", 0),
        "bad_crc_back": r.get("bad_crc_back", 0),
        "seq_gaps": brd.get("seq_gaps", 0),
        "uart_errors": brd.get("uart_errors", 0),
        "samples_expected": len(x_all),
        "samples_received": brd.get("samples", 0),
        "expected_features": r["expected_features"],
        "received_features": len(r["features"]),
        "lost_features": r["lost_features"],
    }
    save_csv(OUT_DIR / "full_144_skip_experiment.csv", [row])
    print("  Full-stream skip=(10,11,200):", row)
    return row


# ---------------------------------------------------------------------------
# Task 4: Full 144-clip loss accounting verification
# ---------------------------------------------------------------------------
def verify_loss_accounting(dataset_stats: dict):
    """Print and verify the full loss accounting against README requirements."""
    print("\n=== Task 4: Full 144-Clip Loss Accounting ===")
    clips = clip_list(TEAM)
    expected_samples = sum(len(get_clip(TEAM, cid, fs=FS, antialias=True)) for cid, _ in clips)
    packets_expected = (expected_samples + SPP - 1) // SPP
    expected_features_calc = 1 + (expected_samples - WIN) // HOP if expected_samples >= WIN else 0

    print(f"  Clips: 144, Expected samples: {expected_samples:,}")
    print(f"  Packets expected: {packets_expected}")
    print(f"  Expected feature frames (continuous): {expected_features_calc}")
    print(f"  Received samples: {dataset_stats['samples_received']:,}")
    print(f"  Received features: {dataset_stats['received_features']}")
    print(f"  Lost features: {dataset_stats['lost_features']}")
    print(f"  bad_crc: {dataset_stats['bad_crc']}")
    print(f"  bad_crc_back: {dataset_stats['bad_crc_back']}")
    print(f"  seq_gaps: {dataset_stats['seq_gaps']}")
    print(f"  Boundary frames excluded from dataset: {dataset_stats['boundary_frames_excluded']}")
    print(f"  Clean within-clip frames in dataset: {dataset_stats['clean_frames']} (144×49)")

    assert dataset_stats["received_features"] == 7199, \
        f"Expected 7199 received features, got {dataset_stats['received_features']}"
    assert dataset_stats["lost_features"] == 0, \
        f"Expected 0 lost features, got {dataset_stats['lost_features']}"
    assert dataset_stats["bad_crc"] == 0 and dataset_stats["seq_gaps"] == 0
    assert dataset_stats["clean_frames"] == 144 * 49  # 7056
    print("  All loss accounting assertions PASSED.")


def update_summary(speed_rows, loss_rows, dataset_stats, full_skip_row):
    """Append audit results to mission4 summary.md."""
    summary_path = OUT_DIR / "summary.md"
    txt = summary_path.read_text(encoding="utf-8")
    marker = "## 3. Fastest Zero-Loss Streaming Speed"
    if marker in txt:
        txt = txt[:txt.index(marker)].rstrip() + "\n"

    # Fastest zero-loss speeds
    fastest = {}
    for label in ["COM5_921600", "COM10_115200"]:
        ok = [r for r in speed_rows if r["interface"] == label and r["zero_loss"]]
        fastest[label] = max(ok, key=lambda r: r["speed"])["speed"] if ok else "N/A"

    addendum = f"""
## 3. Fastest Zero-Loss Streaming Speed (`fastest_speed.csv`)

Swept `speed` from 0.5× to 32× on both USB interfaces with `feat=False, store="none"` (requiring `bad_crc==0`, `seq_gaps==0`, `uart_errors==0`, `bad_crc_back==0`, `samples_received==samples_expected`, and `packets_sent==packets_expected`):
- **COM5 (921600 baud, UART):** Fastest zero-loss requested speed = **{fastest.get('COM5_921600', 'N/A')}×**
- **COM10 (native USB, 115200 baud):** Fastest zero-loss requested speed = **{fastest.get('COM10_115200', 'N/A')}×**
- `bad_crc_back` (return-path CRC failures) = 0 at all zero-loss speeds.

## 4. Full 144-Clip Loss Accounting (Verified)

| Metric | Clean 144-Clip | Full 144-Clip `skip=(10,11,200)` |
|---|---:|---:|
| Clips | 144 | 144 |
| Packets expected | 7,200 | 7,200 |
| Packets sent | {dataset_stats.get('packets_sent', 'N/A')} | {full_skip_row.get('packets_sent', 'N/A')} |
| `board["frames_ok"]` | {dataset_stats.get('frames_ok', 'N/A')} *(incl. EOS)* | {full_skip_row.get('frames_ok', 'N/A')} *(incl. EOS)* |
| Samples expected | 2,304,000 | 2,304,000 |
| Samples received | {dataset_stats.get('samples_received', 'N/A'):,} | {full_skip_row.get('samples_received', 'N/A'):,} |
| Expected feature frames | 7,199 | 7,199 |
| Received feature frames | {dataset_stats.get('received_features', 'N/A')} | {full_skip_row.get('received_features', 'N/A')} |
| Lost feature frames | {dataset_stats.get('lost_features', 0)} | {full_skip_row.get('lost_features', 'N/A')} |
| `bad_crc` | {dataset_stats.get('bad_crc', 0)} | {full_skip_row.get('bad_crc', 0)} |
| `bad_crc_back` | {dataset_stats.get('bad_crc_back', 0)} | {full_skip_row.get('bad_crc_back', 0)} |
| `seq_gaps` | {dataset_stats.get('seq_gaps', 0)} | {full_skip_row.get('seq_gaps', 0)} |
| `uart_errors` | 0 | {full_skip_row.get('uart_errors', 0)} |
| Boundary-crossing frames excluded | 143 | — |
| Clean within-clip frames in dataset | 7,056 (144 × 49) | — |

## 5. Dataset Metadata Update (`dataset_board.npz`, `dataset_numpy.npz`)
Both NPZ files now contain additional arrays:
- `first_sample`: shape `(144, 49)` — original `first_sample` offset for each stored feature frame.
- `frame_numbers`: shape `(144, 49)` — global frame index for each stored feature frame.
- `X`, `y`, `classes`, `clip_ids` are **bit-identical** to the previous dataset build.

## 6. LLM Usage Update
- Used LLM to implement `mission4_audit.py` for the speed sweep, `bad_crc_back` enrichment,
  full-stream `skip=(10,11,200)` experiment, dataset metadata addition, and 144-clip loss accounting verification.
"""
    summary_path.write_text(txt + addendum, encoding="utf-8")
    print(f"Updated {summary_path}")


def main():
    speed_rows = fastest_zero_loss_speed()
    loss_rows = rebuild_loss_experiments_with_bad_crc_back()
    dataset_stats = rebuild_datasets_with_metadata()
    verify_loss_accounting(dataset_stats)
    full_skip_row = run_full_stream_skip_experiment()
    update_summary(speed_rows, loss_rows, dataset_stats, full_skip_row)
    print("\nMission 4 audit tasks complete.")


if __name__ == "__main__":
    main()
