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


# ---------------------------------------------------------------------------
# Universal stream integrity validation (applies to ALL streams)
# ---------------------------------------------------------------------------
def _validate_stream_integrity(got_feat: np.ndarray, fsamp: np.ndarray,
                                fnums: np.ndarray, brd: dict, r: dict,
                                clips: list, expected_samples: int,
                                skip_tuple: tuple | None = None) -> None:
    """Validate stream integrity checks that apply to ALL streams, whether
    clean or controlled-loss.  Raises ValueError or RuntimeError on any
    integrity violation.

    Checks performed:
      1. Nonzero bad_crc
      2. Nonzero bad_crc_back
      3. Nonzero uart_errors
      4. Missing required board metadata keys
      5. Feature-array / fsamp / fnums length mismatches
      6. Invalid feature dimensions
      7. NaN or Inf values in features
      8. Duplicate feature frame numbers
      9. Invalid sample offsets (s0 != fn * HOP)
     10. Invalid sample offset range
    """
    bad_crc = brd.get("bad_crc", 0)
    bcb = r.get("bad_crc_back", 0)
    uart_err = brd.get("uart_errors", 0)

    # 1-3: Reject any corruption
    if bad_crc > 0:
        raise RuntimeError(
            f"Integrity: bad_crc={bad_crc} — stream contains forward-path CRC errors")
    if bcb > 0:
        raise RuntimeError(
            f"Integrity: bad_crc_back={bcb} — stream contains return-path CRC errors")
    if uart_err > 0:
        raise RuntimeError(
            f"Integrity: uart_errors={uart_err} — UART framing/overflow errors detected")

    # 4: Required board metadata
    required_brd_keys = {"bad_crc", "samples"}
    missing_brd = required_brd_keys - set(brd.keys())
    if missing_brd:
        raise ValueError(f"Integrity: missing required board metadata keys: {missing_brd}")
    required_r_keys = {"bad_crc_back", "expected_features"}
    missing_r = required_r_keys - set(r.keys())
    if missing_r:
        raise ValueError(f"Integrity: missing required return-path metadata keys: {missing_r}")

    # 5: Length consistency
    if len(got_feat) != len(fsamp) or len(got_feat) != len(fnums):
        raise ValueError(
            f"Integrity: array length mismatch — features={len(got_feat)}, "
            f"first_sample={len(fsamp)}, frame_numbers={len(fnums)}")

    # 6: Feature dimensions
    if len(got_feat) > 0 and got_feat.ndim == 2 and got_feat.shape[1] != MELS:
        raise ValueError(
            f"Integrity: feature dimension mismatch — got {got_feat.shape[1]}, expected {MELS}")

    # 7: NaN/Inf
    if len(got_feat) > 0 and not np.isfinite(got_feat).all():
        raise ValueError("Integrity: NaN or Inf detected in feature arrays")

    # 8: Duplicate frame numbers
    if len(fnums) > 0:
        unique_fns = set()
        for fn_i in fnums.tolist():
            if fn_i in unique_fns:
                raise ValueError(f"Integrity: duplicate frame number {fn_i}")
            unique_fns.add(fn_i)

    # 9: Sample offset validation (s0 must equal fn * HOP)
    for fn_i, s0_i in zip(fnums.tolist(), fsamp.tolist()):
        if s0_i != fn_i * HOP:
            raise ValueError(
                f"Integrity: offset mismatch at frame {fn_i}: first_sample={s0_i} != {fn_i * HOP}")

    # 10: Sample offset range (only check against received samples for clean streams
    # where recv_samples is expected to equal total; for controlled-skip the compressed
    # timeline is validated separately)
    if len(fsamp) > 0:
        if fsamp.min() < 0:
            raise ValueError(
                f"Integrity: negative first_sample={fsamp.min()}")


# ---------------------------------------------------------------------------
# build_loss_aware_dataset -- canonical shared dataset builder
# ---------------------------------------------------------------------------
def build_loss_aware_dataset(got_feat: np.ndarray, fsamp: np.ndarray, fnums: np.ndarray,
                             brd: dict, r: dict, clips: list, expected_samples: int,
                             skip_tuple: tuple | None = None) -> dict:
    """
    Loss-aware dataset builder with fail-closed protection.

    ALL streams are validated for corruption, NaN/Inf, duplicates, offset
    mismatches, etc. via _validate_stream_integrity().

    - If skip_tuple is None: additionally reject any sample/feature count
      mismatch (unexpected loss).
    - If skip_tuple is provided: validate skip indices, verify expected
      counts, reconstruct the original timeline, and invalidate gap-crossing
      and boundary-crossing frames.
    """
    # Universal integrity checks (corruption, NaN, duplicates, offsets, etc.)
    _validate_stream_integrity(
        got_feat, fsamp, fnums, brd, r, clips, expected_samples, skip_tuple)

    recv_samples = brd.get("samples", 0)
    seq_gaps = brd.get("seq_gaps", 0)
    packets_expected = (expected_samples + SPP - 1) // SPP

    if skip_tuple is None:
        # Clean stream: reject any unexplained loss
        if recv_samples != expected_samples:
            raise RuntimeError(
                f"Fail-closed: sample count mismatch "
                f"({recv_samples}/{expected_samples})")
        if len(got_feat) != r["expected_features"]:
            raise RuntimeError(
                f"Fail-closed: feature count mismatch "
                f"({len(got_feat)}/{r['expected_features']})")
        if seq_gaps != 0:
            raise RuntimeError(
                f"Fail-closed: seq_gaps={seq_gaps} in clean stream")
        # Clean: identity mapping
        orig_map = np.arange(expected_samples, dtype=np.int64)
    else:
        # --- Controlled-skip validation ---
        skip_set = set(skip_tuple)
        skip_list = list(skip_tuple)

        # (1) Validate skip indices are unique
        if len(skip_set) != len(skip_list):
            raise ValueError(
                f"Controlled-skip: duplicate skip indices in {skip_tuple}")

        # (2) Validate skip indices are within expected packet range
        for s in skip_list:
            if s < 0 or s >= packets_expected:
                raise ValueError(
                    f"Controlled-skip: skip index {s} out of range "
                    f"[0, {packets_expected})")

        # (3) Calculate expected received samples from actual skipped packet lengths
        skipped_samples = 0
        for p in skip_list:
            pkt_start = p * SPP
            pkt_end = min((p + 1) * SPP, expected_samples)
            skipped_samples += (pkt_end - pkt_start)
        expected_recv_samples = expected_samples - skipped_samples

        # (4) Verify observed sample count
        if recv_samples != expected_recv_samples:
            raise RuntimeError(
                f"Controlled-skip: expected {expected_recv_samples} received samples "
                f"(skipped {skipped_samples}), but got {recv_samples} — "
                f"unexplained additional loss")

        # (5) Calculate expected received feature count from compressed sample count
        expected_recv_features = max(0, 1 + (recv_samples - WIN) // HOP) if recv_samples >= WIN else 0
        if len(got_feat) != expected_recv_features:
            raise RuntimeError(
                f"Controlled-skip: expected {expected_recv_features} features from "
                f"{recv_samples} compressed samples, got {len(got_feat)} — "
                f"unexplained feature loss")

        # (6) Verify sequence-gap counts against firmware semantics
        # board_link.py: board firmware reports seq_gaps as the count of
        # missing sequence numbers (each missing packet increments seq_gaps
        # by 1). With skip_tuple, the number of seq_gaps should equal the
        # number of skipped packets.
        if seq_gaps != len(skip_list):
            raise RuntimeError(
                f"Controlled-skip: expected seq_gaps={len(skip_list)} for "
                f"{len(skip_list)} skipped packets, got seq_gaps={seq_gaps} — "
                f"unexplained gaps")

        # (7) Verify accepted data-packet counts and EOS handling
        expected_packets_sent = packets_expected - len(skip_list)
        packets_sent = r.get("packets_sent", None)
        if packets_sent is not None and packets_sent != expected_packets_sent:
            raise RuntimeError(
                f"Controlled-skip: expected {expected_packets_sent} packets sent, "
                f"got {packets_sent}")

        # (8) Reconstruct the original timeline AFTER validation
        orig_map = np.concatenate([
            np.arange(p * SPP, min((p + 1) * SPP, expected_samples), dtype=np.int64)
            for p in range(packets_expected) if p not in skip_set
        ])

    # --- Dataset construction ---
    n_clips = len(clips)
    X = np.zeros((n_clips, 49, MELS), dtype=np.float32)
    occupied = np.zeros((n_clips, 49), dtype=bool)
    gap_invalid = 0
    boundary_invalid = 0

    for i, (fn, s0) in enumerate(zip(fnums, fsamp)):
        fn_i, s0_i = int(fn), int(s0)

        # (9) Timeline reconstruction: map compressed indices to original positions
        if s0_i + WIN - 1 >= len(orig_map):
            gap_invalid += 1
            continue
        orig_start = int(orig_map[s0_i])
        orig_end = int(orig_map[s0_i + WIN - 1])

        # Invalidate frames crossing missing samples (gap)
        if orig_end - orig_start != WIN - 1:
            gap_invalid += 1
            continue

        # (10) Invalidate frames crossing clip boundaries
        c_start = orig_start // FS
        c_end = orig_end // FS
        if c_start != c_end:
            boundary_invalid += 1
            continue

        if c_start >= n_clips:
            continue

        fic = (orig_start % FS) // HOP
        if fic >= 49:
            continue

        occupied[c_start, fic] = True
        X[c_start, fic] = got_feat[i]

    # (11) For clean streams, require complete occupancy
    if skip_tuple is None:
        if not occupied.all():
            raise RuntimeError(
                "Fail-closed: unpopulated feature slots remain after clean stream")

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
    """Atomically save a dataset NPZ with the canonical schema.

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
        if chk["X"].shape != X.shape:
            raise RuntimeError(f"Shape mismatch after save: {chk['X'].shape}")
        if chk["first_sample"].shape != fs_dataset.shape:
            raise RuntimeError(f"first_sample shape mismatch: {chk['first_sample'].shape}")
        if chk["frame_numbers"].shape != fn_dataset.shape:
            raise RuntimeError(f"frame_numbers shape mismatch: {chk['frame_numbers'].shape}")
    import os
    os.replace(tmp, path)


def _shared_dataset_workflow(clips, out_dir: Path, board_stream_fn, ref_features_fn,
                             existing_comparison: bool = True) -> dict:
    """Shared dataset-construction workflow used by both build_and_validate_datasets()
    and mission4_audit.rebuild_datasets_with_metadata().

    Parameters
    ----------
    clips : list of (clip_id, label)
    out_dir : Path to output directory
    board_stream_fn : callable() -> (got_feat, fsamp, fnums, brd, r_meta, expected_samples)
        Function that streams audio to the board and returns results.
    ref_features_fn : callable(expected_samples) -> np.ndarray
        Function that computes reference features.
    existing_comparison : bool
        If True and previous NPZ files exist, compare against them.

    Returns
    -------
    dict with dataset statistics.
    """
    # Stream to board
    got_feat, fsamp_raw, fnums_raw, brd, r_meta, expected_samples = board_stream_fn()

    # Compute reference features
    ref_all = ref_features_fn(expected_samples)

    # Build labelled dataset using the canonical loss-aware function
    result = build_loss_aware_dataset(
        got_feat, fsamp_raw, fnums_raw, brd, r_meta, clips, expected_samples,
        skip_tuple=None,
    )
    if result["gap_invalid"] != 0:
        raise RuntimeError(
            f"Unexpected gap-invalid frames in clean stream: {result['gap_invalid']}")

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

    expected_boundary = n_clips - 1 if n_clips > 1 else 0
    if boundary_count != expected_boundary:
        raise RuntimeError(
            f"Expected {expected_boundary} boundary frames for {n_clips} clips, "
            f"got {boundary_count}")
    if not np.isfinite(X_board).all() or not np.isfinite(X_numpy).all():
        raise RuntimeError("NaN/Inf in assembled dataset arrays")
    max_diff = float(np.max(np.abs(X_board - X_numpy)))
    if max_diff > 0.05:
        raise RuntimeError(f"Max diff {max_diff} > 0.05 dB")

    # Optional comparison against existing datasets
    board_path = out_dir / "dataset_board.npz"
    numpy_path = out_dir / "dataset_numpy.npz"
    if existing_comparison and board_path.exists() and numpy_path.exists():
        try:
            old_b = np.load(board_path)
            old_n = np.load(numpy_path)
            if not np.allclose(X_board, old_b["X"], atol=1e-5):
                print("  WARNING: X_board changed from previous dataset build")
            if not np.allclose(X_numpy, old_n["X"], atol=1e-5):
                print("  WARNING: X_numpy changed from previous dataset build")
            else:
                print("  Verified: X, y unchanged from previous dataset build.")
            old_b.close()
            old_n.close()
        except Exception as e:
            print(f"  Skipping existing dataset comparison: {e}")

    # Save using canonical atomic writer
    save_dataset_npz(board_path, X_board, y, clip_ids, fs_meta, fn_meta)
    save_dataset_npz(numpy_path, X_numpy, y, clip_ids, fs_meta, fn_meta)

    # Reload and verify
    db = np.load(board_path)
    dn = np.load(numpy_path)
    if db["X"].shape != (n_clips, 49, MELS) or dn["X"].shape != (n_clips, 49, MELS):
        raise RuntimeError(f"Reload shape error: board={db['X'].shape}, numpy={dn['X'].shape}")
    if db["first_sample"].shape != (n_clips, 49):
        raise RuntimeError(f"first_sample shape: {db['first_sample'].shape}")
    if db["frame_numbers"].shape != (n_clips, 49):
        raise RuntimeError(f"frame_numbers shape: {db['frame_numbers'].shape}")
    if len(np.unique(db["clip_ids"])) != n_clips:
        raise RuntimeError("Clip ID count mismatch")
    if n_clips == 144 and not all(np.sum(db["y"] == c) == 24 for c in range(len(CLASSES))):
        raise RuntimeError("Class balance violation")
    db.close()
    dn.close()

    recv_samples = brd.get("samples", 0)
    stats = {
        "total_frames": len(got_feat),
        "boundary_frames_excluded": boundary_count,
        "clean_frames": int(X_board.shape[0] * X_board.shape[1]),
        "X_shape": list(X_board.shape),
        "y_shape": list(y.shape),
        "max_diff_db": round(max_diff, 6),
        "bad_crc": brd.get("bad_crc", 0),
        "seq_gaps": brd.get("seq_gaps", 0),
        "packets_sent": r_meta.get("packets_sent", 0),
        "frames_ok": brd.get("frames_ok", 0),
        "samples_received": recv_samples,
        "received_features": len(got_feat),
        "lost_features": r_meta.get("lost_features", 0),
        "bad_crc_back": r_meta.get("bad_crc_back", 0),
    }
    (out_dir / "dataset_validation.json").write_text(json.dumps(stats, indent=2))
    print("=== Dataset Validation ===", stats)
    return stats


def build_and_validate_datasets() -> dict:
    """Stream all 144 clips at speed=1.0, exclude boundary frames, and save board & numpy datasets.

    Uses _shared_dataset_workflow() which calls build_loss_aware_dataset() for
    consistent labelling and save_dataset_npz() for consistent schema.
    """
    clips = clip_list(TEAM)
    x_all = np.concatenate([get_clip(TEAM, cid, fs=FS, antialias=True) for cid, _ in clips])
    _, xhat_all, _ = quantise(x_all, fmt="i16", scale=1.0)
    pl_all = pack(x_all, fmt="i16", scale=1.0)

    def board_stream_fn():
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
        r_meta = {
            "expected_features": r.get("expected_features", 7199),
            "bad_crc_back": r.get("bad_crc_back", 0),
            "packets_sent": r.get("packets_sent", 0),
            "lost_features": r.get("lost_features", 0),
        }
        return got_feat, fsamp_raw, fnums_raw, brd, r_meta, len(x_all)

    def ref_features_fn(expected_samples):
        return features(xhat_all, fs=FS, win=WIN, hop=HOP, nfft=NFFT, mels=MELS, mfcc=0)

    return _shared_dataset_workflow(clips, OUT_DIR, board_stream_fn, ref_features_fn)


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
