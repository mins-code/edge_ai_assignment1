"""
mission4_loss_dataset.py -- Mission 4: Loss Accounting and Dataset Construction
Team 02 | AI255TBD Edge AI Module 2
"""

import csv
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from board_link import Board
from team_audio import CLASSES, clip_list, get_clip
from mission2_sampling_storage import quantise, pack
from mission3_features import features

TEAM = 2
PORT = "COM5"
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


def build_loss_aware_dataset(got_feat: np.ndarray, fsamp: np.ndarray, fnums: np.ndarray,
                             brd: dict, r: dict, clips: list, expected_samples: int,
                             skip_tuple: tuple | None = None) -> dict:
    """
    Loss-aware dataset builder with fail-closed protection:
    - If skip_tuple is None (unexpected/undisclosed loss), reject any stream with
      bad_crc, seq_gaps, uart_errors, bad_crc_back, or sample/feature count mismatch.
    - If skip_tuple is explicitly provided, reconstruct original sample timeline,
      invalidate gap-crossing and boundary-crossing frames, and label valid frames.
    """
    bad_crc = brd.get("bad_crc", 0)
    seq_gaps = brd.get("seq_gaps", 0)
    uart_err = brd.get("uart_errors", 0)
    bcb = r.get("bad_crc_back", 0)
    recv_samples = brd.get("samples", 0)

    if skip_tuple is None:
        if (bad_crc > 0 or seq_gaps > 0 or uart_err > 0 or bcb > 0
                or recv_samples != expected_samples or len(got_feat) != r["expected_features"]):
            raise RuntimeError(
                f"Fail-closed: unexplained packet loss or integrity error detected "
                f"(bad_crc={bad_crc}, seq_gaps={seq_gaps}, uart_err={uart_err}, "
                f"bad_crc_back={bcb}, samples={recv_samples}/{expected_samples}, "
                f"features={len(got_feat)}/{r['expected_features']}); refusing to label compressed timeline."
            )
        orig_map = np.arange(expected_samples, dtype=np.int64)
    else:
        packets_expected = (expected_samples + SPP - 1) // SPP
        orig_map = np.concatenate([
            np.arange(p * SPP, min((p + 1) * SPP, expected_samples), dtype=np.int64)
            for p in range(packets_expected) if p not in skip_tuple
        ])

    n_clips = len(clips)
    X = np.zeros((n_clips, 49, MELS), dtype=np.float32)
    occupied = np.zeros((n_clips, 49), dtype=bool)
    seen = set()
    gap_invalid = 0
    boundary_invalid = 0

    for i, (fn, s0) in enumerate(zip(fnums, fsamp)):
        fn_i, s0_i = int(fn), int(s0)
        if fn_i in seen:
            raise ValueError(f"Duplicate frame number {fn_i}")
        seen.add(fn_i)
        if s0_i != fn_i * HOP:
            raise ValueError(f"Offset mismatch at frame {fn_i}: {s0_i} != {fn_i * HOP}")
        orig_start = int(orig_map[s0_i])
        orig_end = int(orig_map[s0_i + WIN - 1])
        if orig_end - orig_start != WIN - 1:
            gap_invalid += 1
            continue
        c_start = orig_start // FS
        c_end = orig_end // FS
        if c_start != c_end:
            boundary_invalid += 1
            continue
        fic = (orig_start % FS) // HOP
        occupied[c_start, fic] = True
        X[c_start, fic] = got_feat[i]

    return {
        "X": X,
        "occupied": occupied,
        "gap_invalid": gap_invalid,
        "boundary_invalid": boundary_invalid,
    }


def run_loss_experiments() -> list[dict]:
    """Run control and deliberate packet-loss experiments on 6s of Team 02 audio."""
    x = np.concatenate([get_clip(TEAM, i, fs=FS, antialias=True) for i in range(6)])
    pl = pack(x, fmt="i16", scale=1.0)
    samples_expected = len(x)
    packets_expected = (samples_expected + SPP - 1) // SPP  # 300 packets for 6s

    experiments = [
        ("control_no_loss", ()),
        ("readme_skip_10_11_200", (10, 11, 200)),
        ("single_beginning_5", (5,)),
        ("burst_20_to_23", (20, 21, 22, 23)),
    ]
    rows = []
    for name, skip_tuple in experiments:
        b = Board(PORT, baud=BAUD)
        try:
            r = b.stream(pl, fs=FS, fmt="i16", scale=1.0, win=WIN, hop=HOP,
                         nfft=NFFT, mels=MELS, mfcc=0, feat=True, store="none",
                         speed=1.0, samples_per_packet=SPP, skip=skip_tuple)
        finally:
            b.close()
        brd = r["board"] or {}
        fsamp = np.asarray(r["first_sample"], dtype=np.int64)
        fnums = np.asarray(r["frame_numbers"], dtype=np.int64)
        # Check shift of last frame's first_sample relative to original unskipped timeline
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
        print(row)

    save_csv(OUT_DIR / "loss_experiments.csv", rows)

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    names = [r["experiment"] for r in rows]
    ax.bar(names, [r["lost_features"] for r in rows], color="tab:red", alpha=0.8, label="Lost Feature Frames")
    ax.set_ylabel("Lost Feature Frames")
    ax.set_title("Mission 4: Deliberate Packet Loss vs Lost Feature Frames")
    ax.tick_params(axis="x", rotation=15)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "loss_comparison.png", dpi=150)
    plt.close(fig)
    return rows


def save_dataset_npz(
    path: Path,
    X: np.ndarray,
    y: np.ndarray,
    clip_ids: np.ndarray,
    fs_dataset: np.ndarray,
    fn_dataset: np.ndarray,
) -> None:
    """Atomically save a dataset NPZ with the canonical schema (Bug 3 fix).

    Schema:
      X            float32 (n_clips, 49, MELS)
      y            int64   (n_clips,)
      clip_ids     int64   (n_clips,)
      classes      str_    (n_classes,)
      first_sample int64   (n_clips, 49)
      frame_numbers int64  (n_clips, 49)
      Scalar metadata: fs, win, hop, nfft, mels, fmt, scale
    """
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        tmp,
        X=X, y=y, clip_ids=clip_ids, classes=np.array(CLASSES),
        first_sample=fs_dataset, frame_numbers=fn_dataset,
        fs=FS, win=WIN, hop=HOP, nfft=NFFT, mels=MELS, fmt="i16", scale=1.0,
    )
    # Verify before promoting — use allow_pickle=False and explicit close for Windows
    with np.load(tmp, allow_pickle=False) as chk:
        assert chk["X"].shape == X.shape, f"Shape mismatch after save: {chk['X'].shape}"
        assert chk["first_sample"].shape == fs_dataset.shape
        assert chk["frame_numbers"].shape == fn_dataset.shape
    import os
    os.replace(tmp, path)


def build_and_validate_datasets() -> dict:
    """Stream all 144 clips at speed=1.0, exclude boundary frames, and save board & numpy datasets.

    Uses build_loss_aware_dataset() for consistent labelling (Bug 1 fix).
    Saves both datasets with save_dataset_npz() for consistent schema (Bug 3 fix).
    """
    clips = clip_list(TEAM)
    x_all = np.concatenate([get_clip(TEAM, cid, fs=FS, antialias=True) for cid, _ in clips])
    _, xhat_all, _ = quantise(x_all, fmt="i16", scale=1.0)
    pl_all = pack(x_all, fmt="i16", scale=1.0)
    ref_all = features(xhat_all, fs=FS, win=WIN, hop=HOP, nfft=NFFT, mels=MELS, mfcc=0)

    b = Board(PORT, baud=BAUD)
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
    expected_samples = len(x_all)

    # Pre-call validation
    assert brd is not None, "Board summary is None"
    assert len(got_feat) == len(fsamp_raw) == len(fnums_raw), (
        f"Length mismatch: feats={len(got_feat)}, fsamp={len(fsamp_raw)}, fnums={len(fnums_raw)}")
    assert got_feat.shape[1] == MELS, f"Feature dim mismatch: {got_feat.shape[1]} != {MELS}"
    assert np.isfinite(got_feat).all(), "NaN/Inf in board features"
    assert fsamp_raw.min() >= 0 and fsamp_raw.max() < expected_samples, \
        f"first_sample out of range: [{fsamp_raw.min()}, {fsamp_raw.max()}]"

    # Additional stream integrity checks
    r_check = {
        "expected_features": r.get("expected_features", 7199),
        "bad_crc_back": r.get("bad_crc_back", 0),
    }

    # Build labelled dataset using the canonical loss-aware function (Bug 1 fix)
    result = build_loss_aware_dataset(
        got_feat, fsamp_raw, fnums_raw, brd, r_check, clips, expected_samples,
        skip_tuple=None,
    )
    assert result["gap_invalid"] == 0, \
        f"Unexpected gap-invalid frames in clean stream: {result['gap_invalid']}"
    assert result["occupied"].all(), "Unpopulated slots remain after clean stream"

    X_board = result["X"]
    occupied = result["occupied"]
    boundary_count = result["boundary_invalid"]

    # Build numpy reference features aligned to same clip/frame slots
    n_clips = len(clips)
    X_numpy = np.zeros((n_clips, 49, MELS), dtype=np.float32)
    fs_meta = np.zeros((n_clips, 49), dtype=np.int64)
    fn_meta = np.zeros((n_clips, 49), dtype=np.int64)

    for i, (fn_i, s0_i) in enumerate(zip(fnums_raw.tolist(), fsamp_raw.tolist())):
        c_start = s0_i // FS
        c_end = (s0_i + WIN - 1) // FS
        if c_start != c_end:
            continue
        frame_in_clip = (s0_i % FS) // HOP
        X_numpy[c_start, frame_in_clip] = ref_all[fn_i]
        fs_meta[c_start, frame_in_clip] = s0_i
        fn_meta[c_start, frame_in_clip] = fn_i

    y = np.array([CLASSES.index(lab) for _, lab in clips], dtype=np.int64)
    clip_ids = np.array([cid for cid, _ in clips], dtype=np.int64)

    assert boundary_count == 143, f"Expected 143 boundary frames, got {boundary_count}"
    assert np.isfinite(X_board).all() and np.isfinite(X_numpy).all()
    max_diff = float(np.max(np.abs(X_board - X_numpy)))
    assert max_diff <= 0.05, f"Max diff {max_diff} > 0.05 dB"

    # Atomic save with consistent schema (Bug 3 fix)
    save_dataset_npz(OUT_DIR / "dataset_board.npz", X_board, y, clip_ids, fs_meta, fn_meta)
    save_dataset_npz(OUT_DIR / "dataset_numpy.npz", X_numpy, y, clip_ids, fs_meta, fn_meta)

    # Reload and verify
    db = np.load(OUT_DIR / "dataset_board.npz")
    dn = np.load(OUT_DIR / "dataset_numpy.npz")
    assert db["X"].shape == (144, 49, 40) and dn["X"].shape == (144, 49, 40)
    assert db["first_sample"].shape == (144, 49)
    assert db["frame_numbers"].shape == (144, 49)
    assert len(np.unique(db["clip_ids"])) == 144
    assert all(np.sum(db["y"] == c) == 24 for c in range(len(CLASSES)))

    stats = {
        "total_frames": len(got_feat),
        "boundary_frames_excluded": boundary_count,
        "clean_frames": int(X_board.shape[0] * X_board.shape[1]),
        "X_shape": list(X_board.shape),
        "y_shape": list(y.shape),
        "max_diff_db": round(max_diff, 6),
        "bad_crc": brd.get("bad_crc", 0),
        "seq_gaps": brd.get("seq_gaps", 0),
        "packets_sent": r.get("packets_sent", 0),
        "frames_ok": brd.get("frames_ok", 0),
        "samples_received": brd.get("samples", 0),
        "received_features": len(got_feat),
        "lost_features": r.get("lost_features", 0),
        "bad_crc_back": r.get("bad_crc_back", 0),
    }
    (OUT_DIR / "dataset_validation.json").write_text(json.dumps(stats, indent=2))
    print("=== Dataset Validation ===", stats)
    return stats


def run_multiclass_loss_alignment() -> dict:
    """Stream 6 clips (one per class) with skip=(10, 49, 150) and reconstruct original timeline."""
    sel_clips = [(c * 24, CLASSES[c]) for c in range(len(CLASSES))]
    x = np.concatenate([get_clip(TEAM, cid, fs=FS, antialias=True) for cid, _ in sel_clips])
    _, xhat, _ = quantise(x, fmt="i16", scale=1.0)
    pl = pack(x, fmt="i16", scale=1.0)
    ref = features(xhat, fs=FS, win=WIN, hop=HOP, nfft=NFFT, mels=MELS, mfcc=0)

    skip_tuple = (10, 49, 150)  # pkt 49 is right at the on->off clip boundary (pkt 50)
    packets_expected = (len(x) + SPP - 1) // SPP
    orig_map = np.concatenate([
        np.arange(p * SPP, min((p + 1) * SPP, len(x)))
        for p in range(packets_expected) if p not in skip_tuple
    ])

    b = Board(PORT, baud=BAUD)
    try:
        r = b.stream(pl, fs=FS, fmt="i16", scale=1.0, win=WIN, hop=HOP,
                     nfft=NFFT, mels=MELS, mfcc=0, feat=True, store="none",
                     speed=1.0, samples_per_packet=SPP, skip=skip_tuple)
    finally:
        b.close()

    got = r["features"]
    fsamp = np.asarray(r["first_sample"], dtype=np.int64)
    frame_rows = []
    valid_errs = []
    gap_invalid = 0
    boundary_invalid = 0
    for i, s0 in enumerate(fsamp):
        orig_start = int(orig_map[s0])
        orig_end = int(orig_map[s0 + WIN - 1])
        is_contiguous = (orig_end - orig_start == WIN - 1)
        crosses_boundary = (orig_start // FS != orig_end // FS)
        valid = is_contiguous and not crosses_boundary
        if not is_contiguous:
            gap_invalid += 1
        elif crosses_boundary:
            boundary_invalid += 1
        c_idx = orig_start // FS
        cid, lab = sel_clips[c_idx]
        err_db = float(np.max(np.abs(got[i] - ref[orig_start // HOP]))) if is_contiguous else float("nan")
        if valid:
            valid_errs.append(err_db)
        frame_rows.append({
            "recv_frame_idx": i,
            "compressed_first_sample": int(s0),
            "orig_start": orig_start,
            "orig_end": orig_end,
            "is_contiguous": is_contiguous,
            "crosses_boundary": crosses_boundary,
            "valid": valid,
            "clip_id": cid if valid else -1,
            "label": lab if valid else "INVALID",
            "max_abs_err_db": round(err_db, 6) if np.isfinite(err_db) else "",
        })
    save_csv(OUT_DIR / "multiclass_loss_frames.csv", frame_rows)
    summary = {
        "skip": str(skip_tuple),
        "received_frames": len(got),
        "valid_frames": len(valid_errs),
        "gap_invalid_frames": gap_invalid,
        "boundary_invalid_frames": boundary_invalid,
        "max_valid_err_db": round(max(valid_errs), 6),
        "last_compressed_first_sample": int(fsamp[-1]),
        "last_reconstructed_orig_start": int(orig_map[fsamp[-1]]),
        "timeline_shift_samples": int(fsamp[-1] - orig_map[fsamp[-1]]),
    }
    print("=== Multi-Class Loss Alignment ===", summary)
    assert summary["max_valid_err_db"] <= 0.05
    return summary


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["all", "loss_align"], default="all")
    args = ap.parse_args()
    if args.mode == "loss_align":
        run_multiclass_loss_alignment()
        return
    print("=== 1. Deliberate Packet-Loss Experiments ===")
    run_loss_experiments()
    print("\n=== 2. Multi-Class Loss Alignment ===")
    run_multiclass_loss_alignment()
    print("\n=== 3. Full 144-Clip Dataset Construction & Validation ===")
    build_and_validate_datasets()


if __name__ == "__main__":
    main()
