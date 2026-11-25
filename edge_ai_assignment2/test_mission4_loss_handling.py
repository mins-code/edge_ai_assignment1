"""
test_mission4_loss_handling.py -- Regression/integration tests for Mission 4
Team 02 | AI255TBD Edge AI Module 2

Tests cover:
  - build_loss_aware_dataset() clean and loss paths
  - _validate_stream_integrity() universal checks
  - Fail-closed rejection of unexpected errors
  - Controlled skip + unexpected corruption rejection
  - save_dataset_npz() schema correctness
  - Both production entry points via mocked board calls
  - update_summary() Bug 2 fix and idempotency

Run:  python -m pytest test_mission4_loss_handling.py -v
"""

import csv
import json
import types
from pathlib import Path
from unittest.mock import MagicMock, patch, call
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Import under test
# ---------------------------------------------------------------------------
from mission4_loss_dataset import (
    CLASSES, MELS, WIN, HOP, FS, SPP,
    build_loss_aware_dataset,
    save_dataset_npz,
    _validate_stream_integrity,
    _shared_dataset_workflow,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_clean_stream(n_clips=4, frames_per_clip=49):
    """Return (got_feat, fsamp, fnums, brd, r_meta, clips, expected_samples)
    for a perfectly clean stream with n_clips*FS samples total.

    r_meta["expected_features"] is set to len(within-clip frames) so the
    fail-closed check in build_loss_aware_dataset() does NOT trigger.
    """
    total_samples = n_clips * FS  # e.g. 4 s
    frames_total = (total_samples - WIN) // HOP + 1

    # Build continuous fsamp/fnums for all frames
    fnums_all = np.arange(frames_total, dtype=np.int64)
    fsamp_all = fnums_all * HOP

    # Mask out boundary-crossing frames to get "within-clip" frames
    within = []
    for i, s0 in enumerate(fsamp_all):
        c_start = s0 // FS
        c_end = (s0 + WIN - 1) // FS
        if c_start == c_end:
            within.append(i)

    fnums = fnums_all[within]
    fsamp = fsamp_all[within]
    n_recv = len(fnums)
    got_feat = np.ones((n_recv, MELS), dtype=np.float32) * 0.5

    brd = {
        "bad_crc": 0,
        "seq_gaps": 0,
        "uart_errors": 0,
        "samples": total_samples,
        "frames_ok": frames_total + 1,
    }
    # expected_features = len of received (within-clip) features so fail-closed passes
    r_meta = {
        "bad_crc_back": 0,
        "expected_features": n_recv,
    }
    clips = [(c * 24, CLASSES[c % len(CLASSES)]) for c in range(n_clips)]
    return got_feat, fsamp, fnums, brd, r_meta, clips, total_samples


def _make_controlled_skip_stream(n_clips=4, skip_tuple=(10,)):
    """Build a synthetic stream with known skipped packets for controlled-loss testing."""
    total_samples = n_clips * FS
    packets_expected = (total_samples + SPP - 1) // SPP
    skip_set = set(skip_tuple)

    # Compute received samples (compressed timeline)
    orig_map = np.concatenate([
        np.arange(p * SPP, min((p + 1) * SPP, total_samples), dtype=np.int64)
        for p in range(packets_expected) if p not in skip_set
    ])
    recv_samples = len(orig_map)
    frames_recv = max(0, (recv_samples - WIN) // HOP + 1) if recv_samples >= WIN else 0

    fnums = np.arange(frames_recv, dtype=np.int64)
    fsamp = fnums * HOP
    got_feat = np.ones((frames_recv, MELS), dtype=np.float32) * 0.5

    brd = {
        "bad_crc": 0,
        "seq_gaps": len(skip_tuple),
        "uart_errors": 0,
        "samples": recv_samples,
        "frames_ok": frames_recv + 1,
    }
    r_meta = {
        "bad_crc_back": 0,
        "expected_features": frames_recv,
        "packets_sent": packets_expected - len(skip_tuple),
    }
    clips = [(c * 24, CLASSES[c % len(CLASSES)]) for c in range(n_clips)]
    return got_feat, fsamp, fnums, brd, r_meta, clips, total_samples


# ---------------------------------------------------------------------------
# T1: Clean stream dataset construction
# ---------------------------------------------------------------------------
def test_01_clean_stream_no_gaps():
    gf, fs, fn, brd, r, clips, exp_samp = _make_clean_stream(n_clips=4)
    result = build_loss_aware_dataset(gf, fs, fn, brd, r, clips, exp_samp, skip_tuple=None)
    assert result["gap_invalid"] == 0
    assert result["occupied"].all()
    assert result["X"].shape == (4, 49, MELS)
    assert np.isfinite(result["X"]).all()


# ---------------------------------------------------------------------------
# T2: Unexpected seq_gaps rejected (fail-closed)
# ---------------------------------------------------------------------------
def test_02_unexpected_seq_gaps_rejected():
    gf, fs, fn, brd, r, clips, exp_samp = _make_clean_stream()
    brd["seq_gaps"] = 1
    with pytest.raises(RuntimeError, match="Fail-closed"):
        build_loss_aware_dataset(gf, fs, fn, brd, r, clips, exp_samp, skip_tuple=None)


# ---------------------------------------------------------------------------
# T3: Unexpected bad_crc rejected
# ---------------------------------------------------------------------------
def test_03_unexpected_bad_crc_rejected():
    gf, fs, fn, brd, r, clips, exp_samp = _make_clean_stream()
    brd["bad_crc"] = 2
    with pytest.raises(RuntimeError, match="bad_crc"):
        build_loss_aware_dataset(gf, fs, fn, brd, r, clips, exp_samp, skip_tuple=None)


# ---------------------------------------------------------------------------
# T4: Unexpected bad_crc_back rejected
# ---------------------------------------------------------------------------
def test_04_bad_crc_back_rejected():
    gf, fs, fn, brd, r, clips, exp_samp = _make_clean_stream()
    r["bad_crc_back"] = 1
    with pytest.raises(RuntimeError, match="bad_crc_back"):
        build_loss_aware_dataset(gf, fs, fn, brd, r, clips, exp_samp, skip_tuple=None)


# ---------------------------------------------------------------------------
# T5: Unexpected uart_errors rejected
# ---------------------------------------------------------------------------
def test_05_uart_errors_rejected():
    gf, fs, fn, brd, r, clips, exp_samp = _make_clean_stream()
    brd["uart_errors"] = 3
    with pytest.raises(RuntimeError, match="uart_errors"):
        build_loss_aware_dataset(gf, fs, fn, brd, r, clips, exp_samp, skip_tuple=None)


# ---------------------------------------------------------------------------
# T6: Sample count mismatch rejected
# ---------------------------------------------------------------------------
def test_06_sample_count_mismatch_rejected():
    gf, fs, fn, brd, r, clips, exp_samp = _make_clean_stream()
    brd["samples"] = exp_samp - 320   # one missing packet worth of samples
    with pytest.raises(RuntimeError, match="sample count mismatch"):
        build_loss_aware_dataset(gf, fs, fn, brd, r, clips, exp_samp, skip_tuple=None)


# ---------------------------------------------------------------------------
# T7: Feature count mismatch rejected
# ---------------------------------------------------------------------------
def test_07_feature_count_mismatch_rejected():
    gf, fs, fn, brd, r, clips, exp_samp = _make_clean_stream()
    r["expected_features"] = len(gf) + 5   # claim more features were expected
    with pytest.raises(RuntimeError, match="feature count mismatch"):
        build_loss_aware_dataset(gf, fs, fn, brd, r, clips, exp_samp, skip_tuple=None)


# ---------------------------------------------------------------------------
# T8: Duplicate frame numbers rejected
# ---------------------------------------------------------------------------
def test_08_duplicate_frame_numbers_rejected():
    gf, fs, fn, brd, r, clips, exp_samp = _make_clean_stream()
    fn = fn.copy()
    fn[1] = fn[0]   # duplicate
    with pytest.raises(ValueError, match="duplicate frame number"):
        build_loss_aware_dataset(gf, fs, fn, brd, r, clips, exp_samp, skip_tuple=None)


# ---------------------------------------------------------------------------
# T9: Invalid frame offset (s0 != fn * HOP) rejected
# ---------------------------------------------------------------------------
def test_09_invalid_frame_offset_rejected():
    gf, fs, fn, brd, r, clips, exp_samp = _make_clean_stream()
    fs_bad = fs.copy()
    fs_bad[0] = fs[0] + 1   # misaligned
    with pytest.raises(ValueError, match="offset mismatch"):
        build_loss_aware_dataset(gf, fs_bad, fn, brd, r, clips, exp_samp, skip_tuple=None)


# ---------------------------------------------------------------------------
# T10: NaN in features rejected by production validation
# ---------------------------------------------------------------------------
def test_10_nan_in_features_rejected():
    """build_loss_aware_dataset rejects NaN features via _validate_stream_integrity."""
    gf, fs, fn, brd, r, clips, exp_samp = _make_clean_stream()
    gf_nan = gf.copy()
    gf_nan[0, 0] = float("nan")
    with pytest.raises(ValueError, match="NaN or Inf"):
        build_loss_aware_dataset(gf_nan, fs, fn, brd, r, clips, exp_samp, skip_tuple=None)


# ---------------------------------------------------------------------------
# T11: Inf in features rejected by production validation
# ---------------------------------------------------------------------------
def test_11_inf_in_features_rejected():
    """build_loss_aware_dataset rejects Inf features via _validate_stream_integrity."""
    gf, fs, fn, brd, r, clips, exp_samp = _make_clean_stream()
    gf_inf = gf.copy()
    gf_inf[1, 5] = float("inf")
    with pytest.raises(ValueError, match="NaN or Inf"):
        build_loss_aware_dataset(gf_inf, fs, fn, brd, r, clips, exp_samp, skip_tuple=None)


# ---------------------------------------------------------------------------
# T12: Single controlled missing packet — gap_invalid > 0
# ---------------------------------------------------------------------------
def test_12_single_missing_packet_produces_gap_invalid():
    gf, fs, fn, brd, r, clips, exp_samp = _make_controlled_skip_stream(
        n_clips=4, skip_tuple=(5,))
    result = build_loss_aware_dataset(gf, fs, fn, brd, r, clips, exp_samp,
                                       skip_tuple=(5,))
    assert result["gap_invalid"] >= 1, (
        f"Expected gap_invalid >= 1, got {result['gap_invalid']}")


# ---------------------------------------------------------------------------
# T13: Consecutive missing packets — multiple gap_invalid frames
# ---------------------------------------------------------------------------
def test_13_consecutive_missing_packets():
    gf, fs, fn, brd, r, clips, exp_samp = _make_controlled_skip_stream(
        n_clips=4, skip_tuple=(10, 11, 12))
    result = build_loss_aware_dataset(gf, fs, fn, brd, r, clips, exp_samp,
                                       skip_tuple=(10, 11, 12))
    assert result["gap_invalid"] >= 1, (
        f"Expected >= 1 gap_invalid frames from 3-packet burst, got {result['gap_invalid']}")


# ---------------------------------------------------------------------------
# T14: Missing packet near clip boundary
# ---------------------------------------------------------------------------
def test_14_missing_packet_near_clip_boundary():
    """Packet near a 1-second clip boundary means boundary frames AND gap frames both flagged."""
    gf, fs, fn, brd, r, clips, exp_samp = _make_controlled_skip_stream(
        n_clips=4, skip_tuple=(50,))
    result = build_loss_aware_dataset(gf, fs, fn, brd, r, clips, exp_samp,
                                       skip_tuple=(50,))
    assert result["gap_invalid"] >= 1


# ---------------------------------------------------------------------------
# T15 (TEST A): Controlled skip + unexpected CRC — MUST REJECT
# ---------------------------------------------------------------------------
def test_15_controlled_skip_with_unexpected_crc_rejected():
    """A real controlled skip=(10,) with bad_crc=1 must be rejected.
    This tests that integrity checks apply to controlled-skip streams too."""
    gf, fs, fn, brd, r, clips, exp_samp = _make_controlled_skip_stream(
        n_clips=4, skip_tuple=(10,))
    brd["bad_crc"] = 1  # inject unexpected CRC error
    with pytest.raises(RuntimeError, match="bad_crc"):
        build_loss_aware_dataset(gf, fs, fn, brd, r, clips, exp_samp,
                                  skip_tuple=(10,))


# ---------------------------------------------------------------------------
# T16 (TEST B): Controlled skip + return-path corruption — MUST REJECT
# ---------------------------------------------------------------------------
def test_16_controlled_skip_with_bad_crc_back_rejected():
    """Controlled skip=(10,) + bad_crc_back=1 must be rejected."""
    gf, fs, fn, brd, r, clips, exp_samp = _make_controlled_skip_stream(
        n_clips=4, skip_tuple=(10,))
    r["bad_crc_back"] = 1
    with pytest.raises(RuntimeError, match="bad_crc_back"):
        build_loss_aware_dataset(gf, fs, fn, brd, r, clips, exp_samp,
                                  skip_tuple=(10,))


# ---------------------------------------------------------------------------
# T17 (TEST C): Controlled skip + unexpected sample loss — MUST REJECT
# ---------------------------------------------------------------------------
def test_17_controlled_skip_with_unexpected_sample_loss():
    """Known skip but fewer received samples than the skips explain."""
    gf, fs, fn, brd, r, clips, exp_samp = _make_controlled_skip_stream(
        n_clips=4, skip_tuple=(10,))
    # Reduce received samples below what the skip explains
    brd["samples"] = brd["samples"] - 100
    with pytest.raises(RuntimeError, match="unexplained additional loss"):
        build_loss_aware_dataset(gf, fs, fn, brd, r, clips, exp_samp,
                                  skip_tuple=(10,))


# ---------------------------------------------------------------------------
# T18 (TEST D): NaN features into production validation function
# ---------------------------------------------------------------------------
def test_18_nan_features_in_controlled_skip_rejected():
    """NaN features with a controlled skip must also be rejected by the
    production _validate_stream_integrity function."""
    gf, fs, fn, brd, r, clips, exp_samp = _make_controlled_skip_stream(
        n_clips=4, skip_tuple=(10,))
    gf_nan = gf.copy()
    gf_nan[0, 0] = float("nan")
    with pytest.raises(ValueError, match="NaN or Inf"):
        build_loss_aware_dataset(gf_nan, fs, fn, brd, r, clips, exp_samp,
                                  skip_tuple=(10,))


# ---------------------------------------------------------------------------
# T19 (TEST E): Incomplete stream — genuinely fewer frames
# ---------------------------------------------------------------------------
def test_19_incomplete_stream_detected():
    """Remove feature frames from a clean stream so occupancy is incomplete.
    The production function must reject via fail-closed occupancy check."""
    gf, fs, fn, brd, r, clips, exp_samp = _make_clean_stream(n_clips=4)
    # Remove last 10 frames to create genuinely incomplete occupancy
    gf_short = gf[:-10]
    fs_short = fs[:-10]
    fn_short = fn[:-10]
    # Adjust expected_features to match so fail-closed feature count passes,
    # but occupancy will fail
    r["expected_features"] = len(gf_short)
    with pytest.raises(RuntimeError, match="unpopulated"):
        build_loss_aware_dataset(gf_short, fs_short, fn_short, brd, r, clips,
                                  exp_samp, skip_tuple=None)


# ---------------------------------------------------------------------------
# T20 (TEST F): Production integration — both entry points use shared impl
# ---------------------------------------------------------------------------
def test_20_production_integration_shared_implementation(tmp_path):
    """Invoke _shared_dataset_workflow (used by both production entry points)
    with mocked board and verify:
    1. The shared loss-aware function (build_loss_aware_dataset) is called.
    2. The shared NPZ writer (save_dataset_npz) is called.
    3. Both datasets are generated in an initially empty directory.
    4. All required metadata fields are preserved.
    """
    n_clips = 4
    total_samples = n_clips * FS  # e.g. 4 s
    frames_total = (total_samples - WIN) // HOP + 1

    # Build ALL frames (including boundary-crossing ones) so the workflow
    # can count boundary frames correctly
    fnums_all = np.arange(frames_total, dtype=np.int64)
    fsamp_all = fnums_all * HOP
    got_feat = np.ones((frames_total, MELS), dtype=np.float32) * 0.5

    brd = {
        "bad_crc": 0,
        "seq_gaps": 0,
        "uart_errors": 0,
        "samples": total_samples,
        "frames_ok": frames_total + 1,
    }
    r_meta = {
        "bad_crc_back": 0,
        "expected_features": frames_total,
        "packets_sent": (total_samples + SPP - 1) // SPP,
        "lost_features": 0,
    }
    clips = [(c * 24, CLASSES[c % len(CLASSES)]) for c in range(n_clips)]

    # Provide reference features (simple identity)
    ref_all = np.ones((frames_total, MELS), dtype=np.float32) * 0.5

    def mock_board_stream_fn():
        return got_feat, fsamp_all, fnums_all, brd, r_meta, total_samples

    def mock_ref_features_fn(expected_samples):
        return ref_all

    # Track calls by wrapping the actual functions
    import mission4_loss_dataset as m4ld
    original_bla = m4ld.build_loss_aware_dataset
    original_save = m4ld.save_dataset_npz

    bla_calls = []
    save_calls = []

    def tracked_bla(*args, **kwargs):
        bla_calls.append(True)
        return original_bla(*args, **kwargs)

    def tracked_save(*args, **kwargs):
        save_calls.append(True)
        return original_save(*args, **kwargs)

    # Monkeypatch at the module level so _shared_dataset_workflow sees the tracked versions
    old_bla = m4ld.build_loss_aware_dataset
    old_save = m4ld.save_dataset_npz
    try:
        m4ld.build_loss_aware_dataset = tracked_bla
        m4ld.save_dataset_npz = tracked_save
        stats = m4ld._shared_dataset_workflow(
            clips, tmp_path, mock_board_stream_fn, mock_ref_features_fn,
            existing_comparison=False,
        )
    finally:
        m4ld.build_loss_aware_dataset = old_bla
        m4ld.save_dataset_npz = old_save

    # Verify shared functions were called
    assert len(bla_calls) >= 1, "build_loss_aware_dataset was not called"
    assert len(save_calls) >= 2, f"save_dataset_npz called {len(save_calls)} times, expected >= 2"

    # Verify datasets were generated in the directory
    board_path = tmp_path / "dataset_board.npz"
    numpy_path = tmp_path / "dataset_numpy.npz"
    assert board_path.exists(), "dataset_board.npz not created"
    assert numpy_path.exists(), "dataset_numpy.npz not created"

    # Verify metadata fields
    with np.load(board_path, allow_pickle=False) as db:
        required = {"X", "y", "clip_ids", "classes", "first_sample", "frame_numbers",
                     "fs", "win", "hop", "nfft", "mels", "fmt", "scale"}
        assert required.issubset(set(db.keys())), f"Missing keys: {required - set(db.keys())}"
        assert db["X"].shape == (n_clips, 49, MELS)
        assert db["first_sample"].shape == (n_clips, 49)
        assert db["frame_numbers"].shape == (n_clips, 49)

    with np.load(numpy_path, allow_pickle=False) as dn:
        assert required.issubset(set(dn.keys()))

    # Verify validation JSON was written
    validation_json = tmp_path / "dataset_validation.json"
    assert validation_json.exists()
    val_data = json.loads(validation_json.read_text())
    assert "total_frames" in val_data
    assert "max_diff_db" in val_data


# ---------------------------------------------------------------------------
# T21 (TEST G): Missing metadata rejected
# ---------------------------------------------------------------------------
def test_21_missing_board_metadata_rejected():
    """Remove required board metadata and verify explicit rejection."""
    gf, fs, fn, _, r, clips, exp_samp = _make_clean_stream()
    # Empty board dict missing 'bad_crc' and 'samples'
    brd_missing = {}
    with pytest.raises(ValueError, match="missing required board metadata"):
        build_loss_aware_dataset(gf, fs, fn, brd_missing, r, clips, exp_samp,
                                  skip_tuple=None)


def test_21b_missing_return_metadata_rejected():
    """Remove required return-path metadata and verify explicit rejection."""
    gf, fs, fn, brd, _, clips, exp_samp = _make_clean_stream()
    r_missing = {}  # Missing bad_crc_back and expected_features
    with pytest.raises(ValueError, match="missing required return-path metadata"):
        build_loss_aware_dataset(gf, fs, fn, brd, r_missing, clips, exp_samp,
                                  skip_tuple=None)


# ---------------------------------------------------------------------------
# T22 (TEST H): Controlled-skip metadata validation
# ---------------------------------------------------------------------------
def test_22a_duplicate_skip_indices_rejected():
    """Duplicate skip indices must be rejected."""
    gf, fs, fn, brd, r, clips, exp_samp = _make_controlled_skip_stream(
        n_clips=4, skip_tuple=(10,))
    with pytest.raises(ValueError, match="duplicate skip indices"):
        build_loss_aware_dataset(gf, fs, fn, brd, r, clips, exp_samp,
                                  skip_tuple=(10, 10))


def test_22b_out_of_range_skip_indices_rejected():
    """Out-of-range skip indices must be rejected."""
    gf, fs, fn, brd, r, clips, exp_samp = _make_controlled_skip_stream(
        n_clips=4, skip_tuple=(10,))
    packets_expected = (exp_samp + SPP - 1) // SPP
    with pytest.raises(ValueError, match="out of range"):
        build_loss_aware_dataset(gf, fs, fn, brd, r, clips, exp_samp,
                                  skip_tuple=(packets_expected + 100,))


def test_22c_unexpected_seq_gap_count_rejected():
    """Seq_gaps count not matching skip count must be rejected."""
    gf, fs, fn, brd, r, clips, exp_samp = _make_controlled_skip_stream(
        n_clips=4, skip_tuple=(10,))
    brd["seq_gaps"] = 5  # Wrong count — should be 1
    with pytest.raises(RuntimeError, match="unexplained gaps"):
        build_loss_aware_dataset(gf, fs, fn, brd, r, clips, exp_samp,
                                  skip_tuple=(10,))


# ---------------------------------------------------------------------------
# T23: save_dataset_npz writes canonical schema
# ---------------------------------------------------------------------------
def test_23_save_dataset_npz_canonical_schema(tmp_path):
    n = 4
    X = np.random.rand(n, 49, MELS).astype(np.float32)
    y = np.arange(n, dtype=np.int64)
    clip_ids = np.arange(n, dtype=np.int64)
    fs_meta = np.zeros((n, 49), dtype=np.int64)
    fn_meta = np.zeros((n, 49), dtype=np.int64)
    out_path = tmp_path / "test_dataset.npz"
    save_dataset_npz(out_path, X, y, clip_ids, fs_meta, fn_meta)
    with np.load(out_path, allow_pickle=False) as chk:
        assert set(["X", "y", "clip_ids", "classes", "first_sample", "frame_numbers",
                    "fs", "win", "hop", "nfft", "mels", "fmt", "scale"]).issubset(set(chk.keys()))
        assert chk["X"].shape == (n, 49, MELS)
        assert chk["first_sample"].shape == (n, 49)
        assert chk["frame_numbers"].shape == (n, 49)
        assert chk["y"].dtype == np.int64
        assert chk["clip_ids"].dtype == np.int64


# ---------------------------------------------------------------------------
# T24: Both board and numpy paths produce identical schema fields
# ---------------------------------------------------------------------------
def test_24_both_npz_paths_produce_equivalent_schema(tmp_path):
    n = 4
    X_b = np.random.rand(n, 49, MELS).astype(np.float32)
    X_n = np.random.rand(n, 49, MELS).astype(np.float32)
    y = np.arange(n, dtype=np.int64)
    clip_ids = np.arange(n, dtype=np.int64)
    fs_meta = np.ones((n, 49), dtype=np.int64) * 123
    fn_meta = np.ones((n, 49), dtype=np.int64) * 456
    p_b = tmp_path / "board.npz"
    p_n = tmp_path / "numpy.npz"
    save_dataset_npz(p_b, X_b, y, clip_ids, fs_meta, fn_meta)
    save_dataset_npz(p_n, X_n, y, clip_ids, fs_meta, fn_meta)
    with np.load(p_b, allow_pickle=False) as cb, np.load(p_n, allow_pickle=False) as cn:
        assert set(cb.keys()) == set(cn.keys()), "Schema mismatch between board and numpy datasets"
        np.testing.assert_array_equal(cb["first_sample"], cn["first_sample"])
        np.testing.assert_array_equal(cb["frame_numbers"], cn["frame_numbers"])
        np.testing.assert_array_equal(cb["y"], cn["y"])


# ---------------------------------------------------------------------------
# T25: update_summary() uses correct CSV schema (no KeyError 'speed') -- Bug 2
# ---------------------------------------------------------------------------
def test_25_update_summary_no_keyerror(tmp_path):
    """Verify update_summary() doesn't crash with the new CSV schema."""
    import mission4_audit as m4a
    orig_out = m4a.OUT_DIR
    m4a.OUT_DIR = tmp_path
    (tmp_path / "summary.md").write_text("# Mission 4\n", encoding="utf-8")

    speed_rows = [
        {
            "interface": "COM5_921600",
            "requested_speed": 32.0,
            "effective_speed": 2.737,
            "wire_bps": 89361.0,
            "zero_loss": 1,
        },
        {
            "interface": "COM10_115200",
            "requested_speed": 6.0,
            "effective_speed": 5.713,
            "wire_bps": 186536.0,
            "zero_loss": 1,
        },
    ]
    dataset_stats = {
        "packets_sent": 7200, "frames_ok": 7201, "samples_received": 2304000,
        "received_features": 7199, "lost_features": 0, "bad_crc": 0, "bad_crc_back": 0,
        "seq_gaps": 0,
    }
    full_skip_row = {
        "packets_sent": 7197, "frames_ok": 7190, "samples_received": 2302080,
        "received_features": 7196, "lost_features": 3, "bad_crc": 0, "bad_crc_back": 0,
        "seq_gaps": 3, "uart_errors": 0,
    }
    # Must not raise KeyError
    m4a.update_summary(speed_rows, [], dataset_stats, full_skip_row)
    result = (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert "Fastest Zero-Loss" in result
    assert "2.737" in result
    assert "5.713" in result
    m4a.OUT_DIR = orig_out


# ---------------------------------------------------------------------------
# T26: update_summary() is idempotent (no duplicate sections)
# ---------------------------------------------------------------------------
def test_26_update_summary_idempotent(tmp_path):
    import mission4_audit as m4a
    orig_out = m4a.OUT_DIR
    m4a.OUT_DIR = tmp_path
    (tmp_path / "summary.md").write_text("# Mission 4\n", encoding="utf-8")

    speed_rows = [
        {"interface": "COM5_921600", "requested_speed": 32.0, "effective_speed": 2.737,
         "wire_bps": 89361.0, "zero_loss": 1},
    ]
    dataset_stats = {
        "packets_sent": 7200, "frames_ok": 7201, "samples_received": 2304000,
        "received_features": 7199, "lost_features": 0, "bad_crc": 0, "bad_crc_back": 0,
        "seq_gaps": 0,
    }
    full_skip_row = {
        "packets_sent": 7197, "frames_ok": 7190, "samples_received": 2302080,
        "received_features": 7196, "lost_features": 3, "bad_crc": 0, "bad_crc_back": 0,
        "seq_gaps": 3, "uart_errors": 0,
    }
    m4a.update_summary(speed_rows, [], dataset_stats, full_skip_row)
    m4a.update_summary(speed_rows, [], dataset_stats, full_skip_row)  # second call
    text = (tmp_path / "summary.md").read_text(encoding="utf-8")
    count = text.count("## 3. Fastest Zero-Loss Streaming Speed")
    assert count == 1, f"Section appears {count} times; expected exactly 1 (idempotency failure)"
    m4a.OUT_DIR = orig_out


# ---------------------------------------------------------------------------
# T27: Atomic NPZ write — tmp file is cleaned up on success
# ---------------------------------------------------------------------------
def test_27_atomic_npz_no_tmp_residue(tmp_path):
    n = 2
    X = np.zeros((n, 49, MELS), dtype=np.float32)
    y = np.zeros(n, dtype=np.int64)
    clip_ids = np.zeros(n, dtype=np.int64)
    fs_meta = np.zeros((n, 49), dtype=np.int64)
    fn_meta = np.zeros((n, 49), dtype=np.int64)
    out = tmp_path / "atomic_test.npz"
    save_dataset_npz(out, X, y, clip_ids, fs_meta, fn_meta)
    assert out.exists(), "Final NPZ not written"
    tmp_candidate = out.with_suffix(".tmp.npz")
    assert not tmp_candidate.exists(), "Temp file was not cleaned up after atomic write"


# ---------------------------------------------------------------------------
# T28: Inf features with controlled skip rejected
# ---------------------------------------------------------------------------
def test_28_inf_features_in_controlled_skip_rejected():
    """Inf features with a controlled skip must be rejected."""
    gf, fs, fn, brd, r, clips, exp_samp = _make_controlled_skip_stream(
        n_clips=4, skip_tuple=(10,))
    gf_inf = gf.copy()
    gf_inf[0, 0] = float("inf")
    with pytest.raises(ValueError, match="NaN or Inf"):
        build_loss_aware_dataset(gf_inf, fs, fn, brd, r, clips, exp_samp,
                                  skip_tuple=(10,))


# ---------------------------------------------------------------------------
# T29: Controlled skip + uart_errors rejected
# ---------------------------------------------------------------------------
def test_29_controlled_skip_with_uart_errors_rejected():
    """Controlled skip + uart_errors must be rejected."""
    gf, fs, fn, brd, r, clips, exp_samp = _make_controlled_skip_stream(
        n_clips=4, skip_tuple=(10,))
    brd["uart_errors"] = 1
    with pytest.raises(RuntimeError, match="uart_errors"):
        build_loss_aware_dataset(gf, fs, fn, brd, r, clips, exp_samp,
                                  skip_tuple=(10,))
