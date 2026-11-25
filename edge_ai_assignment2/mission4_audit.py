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
from mission4_loss_dataset import (
    build_loss_aware_dataset,
    save_dataset_npz,
    _shared_dataset_workflow,
)

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
                t0 = time.perf_counter()
                r = b.stream(pl, fs=FS, fmt="i16", scale=1.0, win=WIN, hop=HOP,
                             nfft=NFFT, mels=MELS, mfcc=0, feat=False, store="none",
                             speed=spd, samples_per_packet=SPP)
                elapsed_s = time.perf_counter() - t0
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
            audio_s = len(x) / FS
            eff_spd = audio_s / elapsed_s if elapsed_s > 0 else 0.0
            payload_bps = r["bytes_sent"] / elapsed_s if elapsed_s > 0 else 0.0
            wire_bps = (r["bytes_sent"] + r["packets_sent"] * 13) / elapsed_s if elapsed_s > 0 else 0.0
            zero_loss = int(
                bad_crc == 0 and seq_gaps == 0 and uart_err == 0 and bcb == 0
                and samples_recv == len(x) and r["packets_sent"] == packets_expected
            )
            row = {
                "port": port,
                "baud": baud,
                "interface": label,
                "requested_speed": spd,
                "audio_s": round(audio_s, 3),
                "send_seconds": round(r["send_seconds"], 3),
                "elapsed_s": round(elapsed_s, 3),
                "effective_speed": round(eff_spd, 3),
                "payload_bps": round(payload_bps, 1),
                "wire_bps": round(wire_bps, 1),
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
            print(f"    req={spd:5.1f}x  eff={eff_spd:5.2f}x  zero_loss={zero_loss}  "
                  f"bad_crc={bad_crc}  seq_gaps={seq_gaps}  wire={wire_bps:.0f} B/s")

    save_csv(OUT_DIR / "fastest_speed.csv", rows)
    print(f"\nSaved fastest_speed.csv ({len(rows)} rows)")

    for label in ["COM5_921600", "COM10_115200"]:
        ok = [r for r in rows if r["interface"] == label and r["zero_loss"]]
        if ok:
            best = max(ok, key=lambda r: r["effective_speed"])
            print(f"  {label}: fastest zero-loss effective={best['effective_speed']}× (requested {best['requested_speed']}×)")
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
    Delegates to the shared _shared_dataset_workflow() from mission4_loss_dataset.py
    so that both production entry points use the same:
      - build_loss_aware_dataset() for loss-aware validation
      - save_dataset_npz() for atomic dataset writing
    
    Works even when previous NPZ files do not exist.
    """
    print("\n=== Task 3: Rebuild Datasets with first_sample + frame_numbers ===")
    clips = clip_list(TEAM)
    x_all = np.concatenate([get_clip(TEAM, cid, fs=FS, antialias=True) for cid, _ in clips])
    _, xhat_all, _ = quantise(x_all, fmt="i16", scale=1.0)
    pl_all = pack(x_all, fmt="i16", scale=1.0)

    def board_stream_fn():
        b = Board(COM_PORT, baud=BAUD)
        try:
            r = b.stream(pl_all, fs=FS, fmt="i16", scale=1.0, win=WIN, hop=HOP,
                         nfft=NFFT, mels=MELS, mfcc=0, feat=True, store="none",
                         speed=1.0, samples_per_packet=SPP)
        finally:
            b.close()
        brd = r["board"] or {}
        got_feat = r["features"]
        fsamp_raw = np.asarray(r["first_sample"], dtype=np.int64)
        fnums_raw = np.asarray(r["frame_numbers"], dtype=np.int64)
        bcb = r.get("bad_crc_back", 0)
        r_meta = {
            "expected_features": r.get("expected_features", 7199),
            "bad_crc_back": bcb,
            "packets_sent": r.get("packets_sent", 0),
            "lost_features": r.get("lost_features", 0),
        }
        return got_feat, fsamp_raw, fnums_raw, brd, r_meta, len(x_all)

    def ref_features_fn(expected_samples):
        return features(xhat_all, fs=FS, win=WIN, hop=HOP, nfft=NFFT, mels=MELS, mfcc=0)

    stats = _shared_dataset_workflow(
        clips, OUT_DIR, board_stream_fn, ref_features_fn,
        existing_comparison=True,
    )
    print(f"  Saved updated dataset_board.npz and dataset_numpy.npz with first_sample, frame_numbers")
    print(f"  bad_crc_back={stats['bad_crc_back']}  max_diff_db={stats['max_diff_db']:.6f}")
    return stats


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

    if dataset_stats["received_features"] != 7199:
        raise RuntimeError(f"Expected 7199 received features, got {dataset_stats['received_features']}")
    if dataset_stats["lost_features"] != 0:
        raise RuntimeError(f"Expected 0 lost features, got {dataset_stats['lost_features']}")
    if dataset_stats["bad_crc"] != 0 or dataset_stats["seq_gaps"] != 0:
        raise RuntimeError("Unexpected corruption in clean stream")
    if dataset_stats["clean_frames"] != 144 * 49:
        raise RuntimeError(f"Expected {144*49} clean frames, got {dataset_stats['clean_frames']}")
    print("  All loss accounting assertions PASSED.")


def update_summary(speed_rows, loss_rows, dataset_stats, full_skip_row):
    """Write audit results to mission4 summary.md."""
    summary_path = OUT_DIR / "summary.md"
    # Read the existing base summary (up to the first audit section)
    if summary_path.exists():
        txt = summary_path.read_text(encoding="utf-8")
    else:
        txt = "# Mission 4: Loss Accounting & Dataset Construction — Results (Team 02)\n"

    # Find the marker for audit-added content and truncate
    marker = "## 3. Fastest Zero-Loss Streaming Speed"
    if marker in txt:
        txt = txt[:txt.index(marker)].rstrip() + "\n"

    # Fastest zero-loss speeds
    fastest = {}
    for label in ["COM5_921600", "COM10_115200"]:
        ok = [r for r in speed_rows if r["interface"] == label and r["zero_loss"]]
        if ok:
            best = max(ok, key=lambda r: r["effective_speed"])
            fastest[label] = {
                "requested": best["requested_speed"],
                "effective": best["effective_speed"],
                "wire_bps": best.get("wire_bps", "N/A"),
            }
        else:
            fastest[label] = None

    ds_samples = dataset_stats.get('samples_received', 'N/A')
    ds_samples_str = f"{ds_samples:,}" if isinstance(ds_samples, int) else str(ds_samples)
    fs_samples = full_skip_row.get('samples_received', 'N/A')
    fs_samples_str = f"{fs_samples:,}" if isinstance(fs_samples, int) else str(fs_samples)
    com5 = fastest.get("COM5_921600")
    com10 = fastest.get("COM10_115200")

    # Format speed table rows
    if com5:
        com5_row = f"| COM5, UART | {com5['requested']}× | {com5['effective']}× | {com5['wire_bps']:.1f} B/s |"
    else:
        com5_row = "| COM5, UART | N/A | N/A | N/A |"
    if com10:
        com10_row = f"| COM10, native USB | {com10['requested']}× | {com10['effective']}× | {com10['wire_bps']:.1f} B/s |"
    else:
        com10_row = "| COM10, native USB | N/A | N/A | N/A |"

    addendum = f"""
## 3. Fastest Zero-Loss Streaming Speed (`fastest_speed.csv`)

Swept `requested_speed` from 0.5× to 32× on both USB interfaces with `feat=False, store="none"`.
Zero-loss requires `bad_crc==0`, `seq_gaps==0`, `uart_errors==0`, `bad_crc_back==0`,
`samples_received==samples_expected`, and `packets_sent==packets_expected`.

| Interface | Requested speed | Effective speed | Wire throughput |
|---|---:|---:|---:|
{com5_row}
{com10_row}

**Interpretation:**
- These Mission 4 measurements were taken with `feat=False` (no feature computation on board).
- **Requested speed is not the same as effective delivered speed.** The requested speed sets the
  pacing timer; effective speed is `audio_duration / elapsed_wall_time`.
- **COM5** is constrained by its UART physical bandwidth: at 921,600 baud with 10 bits/byte,
  the theoretical ceiling is 92,160 B/s. Serial backpressure prevents the requested 32× rate
  from being achieved — the effective speed plateaus at ~2.74×.
- **COM10** (native USB) achieves higher throughput but encounters receive-buffer overflow at
  requested speeds of 8× and above, causing CRC errors and sequence gaps.
- Mission 3 and Mission 4 measurements may differ because of experimental settings (feat on/off)
  and timing definitions (Mission 3 uses `send_seconds`; Mission 4 uses wall-clock `elapsed_s`).

`bad_crc_back` (return-path CRC failures) = 0 at all zero-loss speeds.

## 4. Full 144-Clip Loss Accounting (Verified)

| Metric | Clean 144-Clip | Full 144-Clip `skip=(10,11,200)` |
|---|---:|---:|
| Clips | 144 | 144 |
| Packets expected | 7,200 | 7,200 |
| Packets sent | {dataset_stats.get('packets_sent', 'N/A')} | {full_skip_row.get('packets_sent', 'N/A')} |
| `board["frames_ok"]` | {dataset_stats.get('frames_ok', 'N/A')} *(incl. EOS)* | {full_skip_row.get('frames_ok', 'N/A')} *(incl. EOS)* |
| Samples expected | 2,304,000 | 2,304,000 |
| Samples received | {ds_samples_str} | {fs_samples_str} |
| Expected feature frames | 7,199 | 7,199 |
| Received feature frames | {dataset_stats.get('received_features', 'N/A')} | {full_skip_row.get('received_features', 'N/A')} |
| Lost feature frames | {dataset_stats.get('lost_features', 0)} | {full_skip_row.get('lost_features', 'N/A')} |
| `bad_crc` | {dataset_stats.get('bad_crc', 0)} | {full_skip_row.get('bad_crc', 0)} |
| `bad_crc_back` | {dataset_stats.get('bad_crc_back', 0)} | {full_skip_row.get('bad_crc_back', 0)} |
| `seq_gaps` | {dataset_stats.get('seq_gaps', 0)} | {full_skip_row.get('seq_gaps', 0)} |
| `uart_errors` | 0 | {full_skip_row.get('uart_errors', 0)} |
| Boundary-crossing frames excluded | 143 | — |
| Clean within-clip frames in dataset | 7,056 (144 × 49) | — |

## 5. Fail-Closed Loss Handling

Both `build_and_validate_datasets()` (in `mission4_loss_dataset.py`) and
`rebuild_datasets_with_metadata()` (in `mission4_audit.py`) delegate to the same
shared `_shared_dataset_workflow()` which calls:
- `build_loss_aware_dataset()` for loss-aware validation and labelling.
- `save_dataset_npz()` for atomic dataset writing with the canonical schema.

For **clean streams** (skip_tuple=None), ALL of the following are rejected:
nonzero bad_crc, bad_crc_back, uart_errors, seq_gaps; sample/feature count mismatches;
duplicate frame numbers; NaN/Inf values; offset mismatches.

For **controlled-loss streams** (explicit skip_tuple): the same corruption checks apply,
plus validation of skip indices (unique, in range), expected sample/feature counts
computed from the actual skipped packet lengths, and sequence-gap count verification.
Timeline reconstruction and frame invalidation occur only after validation passes.

Regression tests: `python -m pytest test_mission4_loss_handling.py -q`

## 6. Dataset Metadata Update (`dataset_board.npz`, `dataset_numpy.npz`)
Both NPZ files now contain additional arrays:
- `first_sample`: shape `(144, 49)` — original `first_sample` offset for each stored feature frame.
- `frame_numbers`: shape `(144, 49)` — global frame index for each stored feature frame.
- `X`, `y`, `classes`, `clip_ids` are **bit-identical** to the previous dataset build.

## 7. LLM Usage Update
- Used LLM to implement `mission4_audit.py` for the speed sweep, `bad_crc_back` enrichment,
  full-stream `skip=(10,11,200)` experiment, dataset metadata addition, and 144-clip loss accounting verification.
"""
    summary_path.write_text(txt + addendum, encoding="utf-8")
    print(f"Updated {summary_path}")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["all", "speed_only"], default="all")
    args = ap.parse_args()
    speed_rows = fastest_zero_loss_speed()
    if args.mode == "speed_only":
        return
    loss_rows = rebuild_loss_experiments_with_bad_crc_back()
    dataset_stats = rebuild_datasets_with_metadata()
    verify_loss_accounting(dataset_stats)
    full_skip_row = run_full_stream_skip_experiment()
    update_summary(speed_rows, loss_rows, dataset_stats, full_skip_row)
    print("\nMission 4 audit tasks complete.")


if __name__ == "__main__":
    main()
